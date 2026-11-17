"""Resubmission experiments with immutable provenance and exact split archives.

Run from the repository root: .venv/bin/python ieee_access_resubmit/experiments.py
All model selection uses validation data. Existing submission outputs are read only.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hybrid_pathloss import load_measurements, split_rows, packet_groups, spatial_groups, make_model, rmse
from revision_experiments import minimum_distances

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'protocol.json').read_text())
CONFIGS = PLAN['configs']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def anchor_design(distance_km, height):
    d = np.log10(np.asarray(distance_km))
    h = np.log10(np.asarray(height))
    return np.column_stack([np.ones(len(d)), d, h, d * h])


def fit_height_anchor(df, train):
    # Each training gateway/distance bin contributes one median, including height.
    bins = df.iloc[train].groupby(['gw', 'gw_distance_rounded_km'], as_index=False).agg(
        path_loss=('path_loss', 'median'), antenna_height=('antenna_height', 'first'),
        distance_km=('gw_distance_m', lambda d: d.median() / 1000))
    design = anchor_design(bins.distance_km, bins.antenna_height)
    coef, _, rank, singular = np.linalg.lstsq(design, bins.path_loss, rcond=None)
    predicted = anchor_design(df.gw_distance_m.to_numpy() / 1000, df.antenna_height) @ coef
    return predicted, {'coefficients': coef.tolist(), 'rank': int(rank), 'training_bins': len(bins),
                       'singular_values': singular.tolist(), 'full_rank': bool(rank == 4)}


def exclude_nearby(df, parts, radius, pairwise=True):
    train, val, test = [np.asarray(p, dtype=int).copy() for p in parts]
    coords = np.deg2rad(df[['latitude', 'longitude']].to_numpy())
    if radius:
        val = val[minimum_distances(coords, val, test) >= radius]
        train = train[minimum_distances(coords, train, test) >= radius]
        if pairwise:
            train = train[minimum_distances(coords, train, val) >= radius]
    return train, val, test


def gateway_split(df, gateway, radius, seed=0):
    idx = np.arange(len(df)); group = packet_groups(df)
    test = idx[df.gw.eq(gateway)]
    other = idx[df.gw.ne(gateway)]
    shared_test = group.iloc[test].isin(set(group.iloc[other])).sum()
    remaining = other[~group.iloc[other].isin(set(group.iloc[test])).to_numpy()]
    after_packets = len(remaining)
    if radius:
        coords = np.deg2rad(df[['latitude', 'longitude']].to_numpy())
        remaining = remaining[minimum_distances(coords, remaining, test) >= radius]
    if len(remaining) < 2 or group.iloc[remaining].nunique() < 2:
        train, val = remaining, np.array([], dtype=int)
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed)
        tr, va = next(splitter.split(remaining, groups=group.iloc[remaining]))
        train, val = remaining[tr], remaining[va]
    return (train, val, test), {'gateway': gateway, 'radius_m': radius,
        'original_development': len(other), 'after_packet_exclusion': after_packets,
        'after_position_exclusion': len(remaining), 'test_shared_proxy_packet_rows_before_exclusion': int(shared_test)}


def matched_sample(original, reference, labels, rng):
    """Match reference stratum counts exactly, sampling from original candidates."""
    original = np.asarray(original, dtype=int); reference = np.asarray(reference, dtype=int)
    if labels is None:
        return np.sort(rng.choice(original, len(reference), replace=False))
    labels = np.asarray(labels)
    unique, counts = np.unique(labels[reference], return_counts=True)
    chosen = []
    for key, count in zip(unique, counts):
        candidates = original[labels[original] == key]
        if len(candidates) < count:
            raise ValueError(f'Insufficient candidates for stratum {key}')
        chosen.extend(rng.choice(candidates, count, replace=False))
    return np.sort(np.asarray(chosen, dtype=int))


def disjoint_spatial_splits(df):
    # Original campaign labels are fixed even if later filtering drops rows.
    codes, unique = pd.factorize(spatial_groups(df), sort=True)
    folds = KFold(PLAN['spatial_folds'], shuffle=True, random_state=PLAN['spatial_partition_seed'])
    idx = np.arange(len(df))
    for fold, (_, test_groups) in enumerate(folds.split(np.arange(len(unique)))):
        test = idx[np.isin(codes, test_groups)]
        remaining = idx[~np.isin(codes, test_groups)]
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.15,
            random_state=PLAN['spatial_partition_seed'] + fold).split(remaining, groups=codes[remaining]))
        yield fold, exclude_nearby(df, (remaining[tr], remaining[va], test), PLAN['spatial_radius_m'])


def score(y, pred):
    return {'rmse_db': rmse(y, pred), 'mae_db': float(mean_absolute_error(y, pred)),
            'r2': float(r2_score(y, pred)) if len(y) > 1 else None}


def fit(df, parts, config, seed, jobs, tune=False):
    train, val, test = parts
    if config not in CONFIGS:
        raise ValueError(config)
    anchor_info = None
    if config == 'ldpl_height':
        base, anchor_info = fit_height_anchor(df, train)
    elif config == 'oh_height':
        base = df.okumura_hata.to_numpy()
    else:
        base = np.zeros(len(df))
    extra = ['antenna_height'] + (['okumura_hata'] if config == 'direct_height_anchor' else [])
    x_tree = df[['latitude', 'longitude', 'gw_distance_rounded_km'] + extra].to_numpy()
    x_knn = df[['latitude', 'longitude', 'log_distance_km'] + extra].to_numpy()
    y = df.path_loss.to_numpy(); target = y - base
    pools = {}; candidates = []
    complexity = PLAN['complexity'] if tune else {'reference': PLAN['complexity']['reference']}
    started = time.perf_counter()
    for algorithm in ['rf', 'xgboost', 'knn']:
        pools[algorithm] = {}
        options = [(str(k), {'n_neighbors': k}) for k in PLAN['knn_candidates'] if k <= len(train)] if algorithm == 'knn' else [(key, value[algorithm]) for key, value in complexity.items()]
        for name, settings in options:
            if algorithm == 'knn':
                model = make_pipeline(StandardScaler(), KNeighborsRegressor(**settings, weights='distance', n_jobs=jobs))
                x = x_knn
            else:
                model = make_model(algorithm, seed, jobs).set_params(**settings)
                x = x_tree
            model.fit(x[train], target[train])
            vp = base[val] + model.predict(x[val])
            tp = base[test] + model.predict(x[test])
            vscore = rmse(y[val], vp)
            pools[algorithm][name] = (vscore, tp, vp)
            params = model[-1].get_params() if algorithm == 'knn' else model.get_params()
            params = {k: v for k, v in params.items() if v is None or isinstance(v, (str, int, float, bool))}
            params = {k: ('NaN' if isinstance(v, float) and np.isnan(v) else v) for k, v in params.items()}
            candidates.append({'algorithm': algorithm, 'candidate': name, 'parameters': params, 'validation_rmse_db': vscore})
            del model
    # Insertion order gives deterministic ties: reference first and ascending k.
    selected = {a: min(pool, key=lambda name: pool[name][0]) for a, pool in pools.items()}
    predictions = {a + '_db': pool[selected[a]][1] for a, pool in pools.items()}
    predictions['ensemble_db'] = np.mean(list(predictions.values()), axis=0)
    validation_prediction = np.mean([pool[selected[a]][2] for a, pool in pools.items()], axis=0)
    sensitivity = {}
    if tune:
        for level in complexity:
            pred = (pools['rf'][level][1] + pools['xgboost'][level][1] + pools['knn'][selected['knn']][1]) / 3
            predictions[level + '_ensemble_db'] = pred
            sensitivity[level] = score(y[test], pred)
    result = {'config': config, 'model_seed': seed, 'tuned': tune, 'selected': selected,
              'validation_rmse_db': rmse(y[val], validation_prediction), 'test': score(y[test], predictions['ensemble_db']),
              'components': {a: score(y[test], predictions[a + '_db']) for a in pools},
              'sensitivity': sensitivity, 'candidates': candidates, 'anchor': anchor_info,
              'seconds': time.perf_counter() - started}
    frame = pd.DataFrame({'source_row': df.iloc[test].index, 'measured_db': y[test], **predictions})
    return result, frame


class Campaign:
    def __init__(self, output, jobs):
        self.output = output; self.jobs = jobs; output.mkdir(parents=True, exist_ok=True)
        paths = ['ieee_access_resubmit/experiments.py', 'ieee_access_resubmit/protocol.json',
                 'hybrid_pathloss.py', 'revision_experiments.py', 'data/samples.csv', 'data/gateways.csv']
        meta = {'sha256': {p: digest(ROOT / p) for p in paths}, 'plan': PLAN, 'jobs': jobs,
                'python': platform.python_version(), 'packages': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'geopy']},
                'machine': platform.platform()}
        manifest = output / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != meta:
            raise RuntimeError('Campaign provenance changed: use a new output directory; do not mix generations.')
        write_json(manifest, meta)
        if not (output / 'source.zip').exists():
            with zipfile.ZipFile(output / 'source.zip', 'x', zipfile.ZIP_DEFLATED) as z:
                for p in paths:
                    if not p.startswith('data/'): z.write(ROOT / p, p)
        self.df, metadata = load_measurements(ROOT / 'data/samples.csv', ROOT / 'data/gateways.csv')
        write_json(output / 'data_audit.json', {**metadata, 'original_paper_raw_rows': 175492,
            'original_paper_filtered_rows': 112372, 'upstream_revision': '74d9e29a55eae6ba0909738e9f623f6e7219e902',
            'snapshot_to_original_filtered_mapping': 'Not identifiable from the released fields; no claim that the original filtering is reproduced.',
            'available_columns': pd.read_csv(ROOT / 'data/samples.csv', nrows=0).columns.tolist(),
            'unavailable_original_filters': ['fewer than five GPS satellites', 'road-snap displacement greater than 20 m', 'altitude above city maximum'],
            'pl_below_fspl_rows': int(self.df.path_loss.lt(self.df.fspl).sum())})
        self.groups = packet_groups(self.df)
        self.tiles = spatial_groups(self.df)
        self.coords = np.deg2rad(self.df[['latitude', 'longitude']].to_numpy())
        self.original = {s: split_rows(self.df, 'spatial_500m', s) for s in PLAN['common_split_seeds']}
        self.buffered = {(s, r): exclude_nearby(self.df, self.original[s], r) for s in PLAN['common_split_seeds'] for r in PLAN['radii_m']}

    def run(self, name, parts, seed, configs=CONFIGS, tune=False, metadata=None):
        folder = self.output / name; folder.mkdir(parents=True, exist_ok=True)
        train, val, test = parts
        feasible = len(train) >= PLAN['minimum_development_rows'] and len(val) >= PLAN['minimum_development_rows'] and len(test) > 1
        detail_path = folder / 'split.json'
        if not detail_path.exists():
            np.savez_compressed(folder / 'assignments.npz', **{key: self.df.iloc[ix].index.to_numpy() for key, ix in zip(['train', 'validation', 'test'], parts)})
            detail = {'name': name, 'model_seed': seed, 'sizes': dict(zip(['train', 'validation', 'test'], map(len, parts))),
                      'feasible': feasible, 'metadata': metadata or {}}
            for a, b, key in [(train, test, 'train_test'), (val, test, 'validation_test'), (train, val, 'train_validation')]:
                dist = minimum_distances(self.coords, a, b)
                detail['minimum_' + key + '_m'] = float(dist.min()) if len(a) and len(b) else None
                detail['shared_packet_groups_' + key] = len(set(self.groups.iloc[a]) & set(self.groups.iloc[b]))
            detail['gateway_counts'] = {key: self.df.iloc[ix].gw.value_counts().to_dict() for key, ix in zip(['train', 'validation', 'test'], parts)}
            detail['occupied_tiles'] = {key: int(self.tiles.iloc[ix].nunique()) for key, ix in zip(['train', 'validation', 'test'], parts)}
            detail['assignments_sha256'] = digest(folder / 'assignments.npz')
            write_json(detail_path, detail)
        if not feasible:
            print('INFEASIBLE', name, len(train), len(val), flush=True); return
        for config in configs:
            destination = folder / (config + '.json')
            if destination.exists():
                existing = json.loads(destination.read_text())
                if digest(folder / (config + '.csv.gz')) != existing['predictions_sha256']:
                    raise RuntimeError('Prediction artifact mismatch: ' + name)
                continue
            result, predictions = fit(self.df, parts, config, seed, self.jobs, tune)
            predpath = folder / (config + '.csv.gz')
            predictions.to_csv(predpath, index=False)
            result.update({'run': name, 'predictions_sha256': digest(predpath), 'split_sha256': digest(folder / 'assignments.npz')})
            write_json(destination, result)
            print(f'{name} {config}: RMSE={result["test"]["rmse_db"]:.4f}; {result["seconds"]:.1f}s', flush=True)
            gc.collect()

    def controls(self):
        for seed in PLAN['common_split_seeds']:
            for radius in PLAN['radii_m']:
                self.run(f'controls/r{radius}_s{seed}', self.buffered[seed, radius], seed,
                         metadata={'radius_m': radius, 'split_seed': seed})

    def tuning(self):
        for seed in PLAN['common_split_seeds']:
            self.run(f'tuning/r500_s{seed}', self.buffered[seed, 500], seed, tune=True,
                     metadata={'radius_m': 500, 'split_seed': seed})

    def counts(self):
        gateway_codes = pd.factorize(self.df.gw)[0]
        tile_codes = pd.factorize(self.tiles)[0]
        combined = pd.factorize(pd.MultiIndex.from_arrays([gateway_codes, tile_codes]))[0]
        for radius in PLAN['radii_m'][1:]:
            for seed in PLAN['common_split_seeds']:
                train, val, test = self.original[seed]; bt, bv, _ = self.buffered[seed, radius]
                for kind, labels in enumerate([None, gateway_codes, combined]):
                    for rep in range(PLAN['subsample_repetitions']):
                        rng = np.random.default_rng(np.random.SeedSequence([20261117, seed, radius, rep, kind]))
                        parts = (matched_sample(train, bt, labels, rng), matched_sample(val, bv, labels, rng), test)
                        method = PLAN['subsample_strata'][kind]
                        self.run(f'counts/{method}_r{radius}_s{seed}_rep{rep}', parts, seed,
                                 configs=PLAN['subsample_configs'], metadata={'radius_m': radius, 'split_seed': seed, 'repetition': rep, 'matching': method})

    def spatial(self):
        for fold, parts in disjoint_spatial_splits(self.df):
            for seed in PLAN['spatial_model_seeds']:
                self.run(f'spatial/fold{fold}_model{seed}', parts, seed,
                         metadata={'fold': fold, 'radius_m': PLAN['spatial_radius_m'], 'partition_seed': PLAN['spatial_partition_seed']})

    def gateways(self):
        for gateway in sorted(self.df.gw.unique()):
            for radius in PLAN['gateway_radii_m']:
                parts, meta = gateway_split(self.df, gateway, radius, PLAN['gateway_seed'])
                self.run(f'gateways/{gateway}_r{radius}', parts, PLAN['gateway_seed'], metadata=meta)

    def quality(self):
        keep = self.df.path_loss.ge(self.df.fspl).to_numpy()
        if keep.all():
            write_json(self.output / 'quality_noop.json', {'filter': 'PL >= FSPL', 'retained': len(keep),
                       'removed': 0, 'reason': 'All released rows already satisfy the reconstructible quality filter; predictions and assignments are unchanged.'})
            return
        for seed in PLAN['common_split_seeds']:
            for radius in [0, 500]:
                parts = tuple(p[keep[p]] for p in self.buffered[seed, radius])
                self.run(f'quality/r{radius}_s{seed}', parts, seed,
                         metadata={'radius_m': radius, 'split_seed': seed, 'removed_total': int((~keep).sum()), 'filter': 'PL >= FSPL'})
        idx = np.arange(len(self.df))
        for gateway in sorted(self.df.gw.unique()):
            test = idx[self.df.gw.eq(gateway)]; remaining = idx[self.df.gw.ne(gateway)]
            tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=0).split(remaining, groups=self.groups.iloc[remaining]))
            parts = tuple(p[keep[p]] for p in (remaining[tr], remaining[va], test))
            self.run(f'quality/gateway_{gateway}', parts, 0, metadata={'gateway': gateway, 'filter': 'PL >= FSPL'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results/v1')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--stages', nargs='+', choices=['controls', 'tuning', 'counts', 'spatial', 'gateways', 'quality'],
                        default=['controls', 'tuning', 'spatial', 'gateways', 'counts', 'quality'])
    args = parser.parse_args()
    campaign = Campaign(args.output, args.jobs)
    for stage in args.stages:
        write_json(args.output / 'status.json', {'state': 'running', 'stage': stage})
        getattr(campaign, stage)()
        write_json(args.output / (stage + '_complete.json'), {'stage': stage, 'complete': True})
    write_json(args.output / 'status.json', {'state': 'completed', 'stages': args.stages})


if __name__ == '__main__':
    main()
