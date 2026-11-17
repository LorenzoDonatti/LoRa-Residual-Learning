"""Matched-information controls and buffered spatial validation for OJAP.

Run with --resume to reuse only artifacts with identical code/data/environment.
All settings are fixed before evaluating test labels. No test-driven selection.
"""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import platform
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from hybrid_pathloss import load_measurements, split_rows, fit_ldpl_bonn_train, make_model, rmse

RADIUS_M = 6371008.8
CONFIGS = {
    'direct_basic': ('zero', []),
    'direct_height': ('zero', ['antenna_height']),
    'direct_anchor': ('zero', ['okumura_hata']),
    'oh_basic': ('okumura_hata', []),
    'oh_height': ('okumura_hata', ['antenna_height']),
    'ldpl_basic': ('ldpl_bonn_train', []),
    'oh_continuous': ('okumura_hata', []),
}


def minimum_distances(coordinates, query, reference):
    if not len(query):
        return np.empty(0)
    if not len(reference):
        return np.full(len(query), np.inf)
    tree = BallTree(coordinates[reference], metric='haversine')
    return tree.query(coordinates[query], k=1)[0][:, 0] * RADIUS_M


def buffered_split(df, seed, buffer_m):
    """Preserve original test tiles; discard nearby development observations.

    Retain validation rows outside the test buffer, then training rows outside
    both test and retained validation buffers. Test rows never change with radius.
    Distances refer to transmitter positions, including across gateways.
    """
    train, validation, test = split_rows(df, 'spatial_500m', seed)
    coordinates = np.deg2rad(df[['latitude', 'longitude']].to_numpy())
    if buffer_m > 0:
        validation = validation[minimum_distances(coordinates, validation, test) >= buffer_m]
        train = train[minimum_distances(coordinates, train, test) >= buffer_m]
        train = train[minimum_distances(coordinates, train, validation) >= buffer_m]
    details = {'train': len(train), 'validation': len(validation), 'test': len(test),
               'buffer_m': buffer_m,
               'min_train_test_m': float(minimum_distances(coordinates, test, train).min()),
               'min_validation_test_m': float(minimum_distances(coordinates, validation, test).min()) if len(validation) else None,
               'min_train_validation_m': float(minimum_distances(coordinates, validation, train).min()) if len(validation) else None}
    return (train, validation, test), details


def fit_config(df, parts, config, seed, jobs):
    train, validation, test = parts
    anchor, extra = CONFIGS[config]
    y = df.path_loss.to_numpy()
    base = np.zeros(len(df)) if anchor == 'zero' else df[anchor].to_numpy()
    target = y - base
    tree_distance = 'gw_distance_km' if config == 'oh_continuous' else 'gw_distance_rounded_km'
    tree_features = ['latitude', 'longitude', tree_distance] + extra
    knn_features = ['latitude', 'longitude', 'log_distance_km'] + extra
    predictions, records = {}, []
    x_tree = df[tree_features].to_numpy()
    x_knn = df[knn_features].to_numpy()
    for algorithm in ('rf', 'xgboost', 'knn'):
        start = time.perf_counter()
        if algorithm == 'knn':
            choices = []
            for k in (5, 10, 20, 50, 100):
                if k > len(train):
                    continue
                model = make_pipeline(StandardScaler(), KNeighborsRegressor(
                    n_neighbors=k, weights='distance', n_jobs=jobs))
                model.fit(x_knn[train], target[train])
                val_pred = base[validation] + model.predict(x_knn[validation])
                choices.append((rmse(y[validation], val_pred), k, model))
            val_rmse, chosen_k, model = min(choices, key=lambda x: (x[0], x[1]))
            x = x_knn
        else:
            chosen_k = None
            x = x_tree
            model = make_model(algorithm, seed, jobs)
            model.fit(x[train], target[train])
            val_rmse = rmse(y[validation], base[validation] + model.predict(x[validation]))
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = base[test] + model.predict(x[test])
        prediction_seconds = time.perf_counter() - start
        predictions[algorithm] = prediction
        records.append({'model': algorithm, 'validation_rmse_db': val_rmse,
                        'test_rmse_db': rmse(y[test], prediction),
                        'selected_k': chosen_k, 'fit_and_validation_seconds': fit_seconds,
                        'test_prediction_seconds': prediction_seconds,
                        'serialized_bytes': len(pickle.dumps(model, protocol=5))})
    predictions['ensemble'] = np.mean(list(predictions.values()), axis=0)
    records.append({'model': 'ensemble', 'test_rmse_db': rmse(y[test], predictions['ensemble']),
                    'fit_and_validation_seconds': sum(r['fit_and_validation_seconds'] for r in records),
                    'test_prediction_seconds': sum(r['test_prediction_seconds'] for r in records),
                    'serialized_bytes': sum(r['serialized_bytes'] for r in records)})
    return records, predictions


def provenance(root, args):
    files = ['revision_experiments.py', 'hybrid_pathloss.py', 'data/samples.csv', 'data/gateways.csv']
    return {'sha256': {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
            'packages': {p: version(p) for p in ('numpy','pandas','scikit-learn','xgboost','geopy')},
            'python': platform.python_version(), 'jobs': args.jobs,
            'settings': {'trees':100,'knn_candidates':[5,10,20,50,100], 'weights':[1/3]*3,
                         'buffers_m':[250,500,1000], 'earth_radius_m': RADIUS_M},
            'machine': platform.platform(), 'logical_cpus': os.cpu_count()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--protocols', nargs='+', default=['random','packet','spatial_500m','buffer250','buffer500','buffer1000'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root/'results/submission_controls'
    output.mkdir(parents=True, exist_ok=True)
    meta = provenance(root, args)
    manifest = output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != meta:
        raise RuntimeError('Provenance changed: use a new output directory or explicitly archive existing results.')
    manifest.write_text(json.dumps(meta, indent=2)+'\n')
    df, _ = load_measurements(root/'data/samples.csv', root/'data/gateways.csv')
    df['gw_distance_km'] = df.gw_distance_m/1000
    for protocol in args.protocols:
        for seed in args.seeds:
            if protocol.startswith('buffer'):
                parts, details = buffered_split(df, seed, int(protocol[6:]))
                configs = ['direct_height','direct_anchor','oh_basic','oh_height']
            else:
                parts = split_rows(df, protocol, seed)
                details = dict(zip(['train','validation','test'], map(len, parts)))
                configs = list(CONFIGS)
            train, validation, test = parts
            (output/f'{protocol}_seed{seed}_split.json').write_text(json.dumps(details, indent=2)+'\n')
            # Report infeasibility rather than silently changing the protocol.
            if len(train) < 100 or len(validation) < 100:
                print(f'INFEASIBLE {protocol} seed{seed}: {details}', flush=True)
                continue
            df['ldpl_bonn_train'], _ = fit_ldpl_bonn_train(df, train)
            for config in configs:
                destination = output/f'{protocol}_seed{seed}_{config}.json'
                prediction_path = output/f'{protocol}_seed{seed}_{config}.csv.gz'
                if args.resume and destination.exists() and prediction_path.exists():
                    continue
                records, predictions = fit_config(df, parts, config, seed, args.jobs)
                result = {'protocol':protocol,'seed':seed,'config':config,'sizes':details,'scores':records}
                artifact = pd.DataFrame({'source_row':df.iloc[test].index, 'measured_db':df.iloc[test].path_loss.to_numpy(),
                                         **{f'{k}_db':v for k,v in predictions.items()}})
                artifact.to_csv(prediction_path, index=False)
                destination.write_text(json.dumps(result, indent=2)+'\n')
                print(f'{protocol} seed{seed} {config}: RMSE={records[-1]["test_rmse_db"]:.4f}, time={records[-1]["fit_and_validation_seconds"]:.1f}s', flush=True)
    rows=[]
    for path in output.glob('*.json'):
        obj=json.loads(path.read_text())
        if 'scores' in obj:
            rows.extend({'protocol':obj['protocol'],'seed':obj['seed'],'config':obj['config'],**r} for r in obj['scores'])
    if rows:
        pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)

if __name__ == '__main__':
    main()
