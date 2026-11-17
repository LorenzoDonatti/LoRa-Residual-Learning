"""Reproducible Bonn LoRa path-loss experiment described in the manuscript.

The public samples.csv already contains the received measurements. Signal-Server
predictions are deliberately excluded because the selected analytical baselines
can be reproduced directly from the two public CSV files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from xgboost import XGBRegressor


FEATURES = ["latitude", "longitude", "gw_distance_rounded_km"]
# Models used as physical priors in the manuscript experiment.
BASELINES = ["fspl", "ldpl_oulu", "okumura_hata"]
# Extra analytical references from the Bonn dataset study. They are evaluated
# as classical baselines but are not automatically added to every hybrid run.
CLASSICAL_BASELINES = ["fspl", "ldpl_oulu", "okumura_hata", "winnerplus"]
FREQUENCY_MHZ = 868.0
SENSOR_HEIGHT_M = 2.0
EIRP_RX_GAIN_LOSS_DB = 14.0 + 3.0 - 2.0


def load_measurements(samples_path: Path, gateways_path: Path) -> tuple[pd.DataFrame, dict]:
    samples = pd.read_csv(samples_path, dtype={"gw": str, "pkt_number": str})
    gateways = pd.read_csv(gateways_path, dtype={"id": str}).set_index("id")
    required_samples = {"latitude", "longitude", "transceived_at", "pkt_number", "rssi", "snr", "gw"}
    required_gateways = {"latitude", "longitude", "antenna_height"}
    if required_samples - set(samples) or required_gateways - set(gateways):
        raise ValueError("Input CSVs lack required Bonn survey columns")
    if gateways.index.has_duplicates:
        raise ValueError("Gateway IDs must be unique")

    raw_rows = len(samples)
    samples = samples.join(
        gateways.rename(columns={"latitude": "gw_latitude", "longitude": "gw_longitude"}),
        on="gw", validate="many_to_one"
    )
    numeric = ["latitude", "longitude", "rssi", "snr", "gw_latitude", "gw_longitude", "antenna_height"]
    samples[numeric] = samples[numeric].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(samples[numeric]).all(axis=1)
    valid &= samples.latitude.between(-90, 90) & samples.gw_latitude.between(-90, 90)
    valid &= samples.longitude.between(-180, 180) & samples.gw_longitude.between(-180, 180)
    valid &= samples.antenna_height.gt(0)
    valid &= samples[["transceived_at", "pkt_number", "gw"]].notna().all(axis=1)
    samples = samples.loc[valid].copy()

    # Round to the nearest metre before forming the ML distance feature, as in
    # the original notebook. Keep metres for the physical formulas.
    samples["gw_distance_m"] = [
        round(geodesic((lat, lon), (gw_lat, gw_lon)).meters)
        for lat, lon, gw_lat, gw_lon in zip(
            samples.latitude, samples.longitude, samples.gw_latitude, samples.gw_longitude
        )
    ]
    samples = samples.loc[samples.gw_distance_m.gt(0)].copy()
    samples["gw_distance_rounded_km"] = (samples.gw_distance_m / 1000).round(2)
    samples["log_distance_km"] = np.log10(samples.gw_distance_m / 1000)
    samples["rpp"] = samples.rssi + np.minimum(samples.snr, 0)
    samples["path_loss"] = EIRP_RX_GAIN_LOSS_DB - samples.rpp

    distance_m = samples.gw_distance_m.to_numpy(dtype=float)
    distance_km = distance_m / 1000
    height_m = samples.antenna_height.to_numpy(dtype=float)
    log_f = np.log10(FREQUENCY_MHZ)
    log_h = np.log10(height_m)
    samples["fspl"] = 20 * np.log10(distance_m) + 20 * log_f - 27.55
    samples["ldpl_oulu"] = 128.95 + 23.2 * np.log10(distance_km)
    receiver_correction = 0.8 + (1.1 * log_f - 0.7) * SENSOR_HEIGHT_M - 1.56 * log_f
    samples["okumura_hata"] = (
        69.55 + 26.16 * log_f - 13.82 * log_h - receiver_correction
        + (44.9 - 6.55 * log_h) * np.log10(distance_km)
    )
    samples["winnerplus"] = (
        (44.9 - 6.55 * log_h) * np.log10(distance_m)
        + 5.83 * log_h + 16.33 + 26.16 * np.log10(FREQUENCY_MHZ / 1000)
    )
    metadata = {
        "source_rows": raw_rows,
        "used_rows": len(samples),
        "excluded_rows": raw_rows - len(samples),
        "target": "15 - (rssi + min(snr, 0)) dB",
        "features": FEATURES,
        "physical_baselines": BASELINES,
        "additional_classical_baselines": [name for name in CLASSICAL_BASELINES if name not in BASELINES],
    }
    return samples, metadata


def fit_ldpl_bonn_train(df: pd.DataFrame, train_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit the Bonn LDPL on training-bin medians only and predict every row.

    This mirrors the source study's equal-bin curve fitting while preventing the
    validation and test targets from determining the local LDPL parameters.
    """
    train = df.iloc[train_idx]
    grouped = train.groupby("gw_distance_rounded_km", as_index=False).agg(
        path_loss=("path_loss", "median")
    )
    grouped = grouped.loc[grouped.gw_distance_rounded_km.gt(0)]
    design = np.column_stack([
        np.ones(len(grouped)), 10 * np.log10(grouped.gw_distance_rounded_km.to_numpy())
    ])
    pl_1km, exponent = np.linalg.lstsq(design, grouped.path_loss.to_numpy(), rcond=None)[0]
    prediction = pl_1km + 10 * exponent * np.log10(df.gw_distance_m.to_numpy() / 1000)
    return prediction, {"pl_at_1km_db": float(pl_1km), "path_loss_exponent": float(exponent),
                        "training_distance_bins": len(grouped)}


def packet_groups(df: pd.DataFrame) -> pd.Series:
    """Proxy transmission key; the public timestamp has date precision only."""
    return (df.transceived_at.astype(str) + "/" + df.pkt_number.astype(str)
            + "/" + df.latitude.astype(str) + "/" + df.longitude.astype(str))


def spatial_groups(df: pd.DataFrame) -> pd.Series:
    """Original 500 m grid, defined on the complete campaign.

    Compute once on the full frame and subset the labels afterwards: recomputing
    the reference latitude on a test subset moves the east/west tile boundaries.
    Tuple labels preserve the ordering used by the original GroupShuffleSplit.
    """
    latitude_m = df.latitude.to_numpy() * 111_320
    longitude_m = df.longitude.to_numpy() * 111_320 * np.cos(np.deg2rad(df.latitude.mean()))
    east = np.floor(longitude_m / 500).astype(int)
    north = np.floor(latitude_m / 500).astype(int)
    return pd.Series(list(zip(east, north)), index=df.index)


def split_rows(df: pd.DataFrame, method: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(df))
    if method == "random":
        train, held = train_test_split(indices, test_size=0.25, random_state=seed)
        validation, test = train_test_split(held, test_size=0.60, random_state=seed)
    elif method in {"packet", "spatial_500m"}:
        # A packet can be received by several gateways; keep those receptions
        # together. Packet IDs can recur within a day, so include GPS position.
        if method == "packet":
            groups = packet_groups(df)
        else:
            groups = spatial_groups(df)
        first = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_pos, held_pos = next(first.split(indices, groups=groups))
        train, held = indices[train_pos], indices[held_pos]
        second = GroupShuffleSplit(n_splits=1, test_size=0.60, random_state=seed)
        val_pos, test_pos = next(second.split(held, groups=groups.iloc[held_pos]))
        validation, test = held[val_pos], held[test_pos]
    else:
        raise ValueError(f"Unknown split method: {method}")
    return train, validation, test


def make_model(name: str, seed: int, jobs: int):
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=jobs)
    if name == "xgboost":
        return XGBRegressor(n_estimators=100, random_state=seed, n_jobs=jobs,
                            objective="reg:squarederror")
    raise ValueError(f"Unknown model: {name}")


def optimal_direct_weight(y: np.ndarray, direct: np.ndarray, hybrid: np.ndarray) -> float:
    """Exact convex weight minimizing validation MSE, hence also RMSE."""
    delta = direct - hybrid
    denominator = float(np.dot(delta, delta))
    if denominator == 0:
        return 0.5
    return float(np.clip(np.dot(y - hybrid, delta) / denominator, 0, 1))


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, prediction)))


def run_experiment(df: pd.DataFrame, split: str, seed: int, jobs: int,
                   models: list[str]) -> tuple[dict, pd.DataFrame]:
    train_idx, val_idx, test_idx = split_rows(df, split, seed)
    x = df[FEATURES].to_numpy()
    y = df.path_loss.to_numpy()
    result: dict = {
        "split": {"method": split, "seed": seed, "train": len(train_idx),
                  "validation": len(val_idx), "test": len(test_idx)},
        "model_settings": {name: settings for name, settings in {
            "rf": {"n_estimators": 100, "random_state": seed},
            "xgboost": {"n_estimators": 100, "max_depth": 6, "random_state": seed},
        }.items() if name in models},
        "classical_test_rmse_db": {}, "direct_test_rmse_db": {},
        "hybrid_test_rmse_db": {},
    }
    predictions = pd.DataFrame({
        "source_row": df.iloc[test_idx].index,
        "gw": df.iloc[test_idx].gw.to_numpy(),
        "distance_km": df.iloc[test_idx].gw_distance_m.to_numpy() / 1000,
        "measured_db": y[test_idx],
    })
    direct_val, hybrid_val = {}, {}
    for baseline in BASELINES:
        result["classical_test_rmse_db"][baseline] = rmse(y[test_idx], df.iloc[test_idx][baseline].to_numpy())
    for name in models:
        model = make_model(name, seed, jobs)
        model.fit(x[train_idx], y[train_idx])
        direct_val[name] = model.predict(x[val_idx])
        pred = model.predict(x[test_idx])
        predictions[f"direct_{name}_db"] = pred
        result["direct_test_rmse_db"][name] = rmse(y[test_idx], pred)
    for baseline in BASELINES:
        base = df[baseline].to_numpy()
        for name in models:
            model = make_model(name, seed, jobs)
            model.fit(x[train_idx], y[train_idx] - base[train_idx])
            key = f"{baseline}_{name}"
            hybrid_val[key] = base[val_idx] + model.predict(x[val_idx])
            pred = base[test_idx] + model.predict(x[test_idx])
            predictions[f"hybrid_{key}_db"] = pred
            result["hybrid_test_rmse_db"][key] = rmse(y[test_idx], pred)

    # This pair is specified by the manuscript. Test labels never select the
    # pair or the weight. The weight is a single global constant.
    if {"rf", "xgboost"}.issubset(models):
        direct_name, hybrid_name = "rf", "okumura_hata_xgboost"
        weight = optimal_direct_weight(y[val_idx], direct_val[direct_name], hybrid_val[hybrid_name])
        test_pred = (weight * predictions[f"direct_{direct_name}_db"].to_numpy()
                     + (1 - weight) * predictions[f"hybrid_{hybrid_name}_db"].to_numpy())
        predictions["ensemble_db"] = test_pred
        result["ensemble"] = {
            "direct": direct_name, "hybrid": hybrid_name,
            "direct_weight": weight, "hybrid_weight": 1 - weight,
            "validation_rmse_db": rmse(y[val_idx], weight * direct_val[direct_name]
                                       + (1 - weight) * hybrid_val[hybrid_name]),
            "test_rmse_db": rmse(y[test_idx], test_pred),
        }
        # Research candidate: equal average of two residual learners sharing
        # the same physical baseline. No validation labels are needed to set
        # its weights; report it separately from the manuscript ensemble.
        rf_hybrid = "okumura_hata_rf"
        xgb_hybrid = "okumura_hata_xgboost"
        average_val = 0.5 * (hybrid_val[rf_hybrid] + hybrid_val[xgb_hybrid])
        average_test = 0.5 * (
            predictions[f"hybrid_{rf_hybrid}_db"].to_numpy()
            + predictions[f"hybrid_{xgb_hybrid}_db"].to_numpy()
        )
        predictions["hybrid_average_db"] = average_test
        result["hybrid_average"] = {
            "purpose": "exploratory comparison; not a manuscript result",
            "components": [rf_hybrid, xgb_hybrid],
            "weights": [0.5, 0.5],
            "validation_rmse_db": rmse(y[val_idx], average_val),
            "test_rmse_db": rmse(y[test_idx], average_test),
        }

    # Exploratory alternative: select both components and the convex weight on
    # validation only. Report it separately from the manuscript's fixed pair.
    candidates = []
    for direct_name, direct_prediction in direct_val.items():
        for hybrid_name, hybrid_prediction in hybrid_val.items():
            weight = optimal_direct_weight(y[val_idx], direct_prediction, hybrid_prediction)
            validation_rmse = rmse(y[val_idx], weight * direct_prediction
                                   + (1 - weight) * hybrid_prediction)
            candidates.append((validation_rmse, direct_name, hybrid_name, weight))
    if candidates:
        validation_rmse, direct_name, hybrid_name, weight = min(candidates)
        test_pred = (weight * predictions[f"direct_{direct_name}_db"].to_numpy()
                     + (1 - weight) * predictions[f"hybrid_{hybrid_name}_db"].to_numpy())
        predictions["validation_selected_ensemble_db"] = test_pred
        result["validation_selected_ensemble"] = {
            "purpose": "exploratory comparison; not a manuscript result",
            "direct": direct_name, "hybrid": hybrid_name,
            "direct_weight": weight, "hybrid_weight": 1 - weight,
            "validation_rmse_db": validation_rmse,
            "test_rmse_db": rmse(y[test_idx], test_pred),
            "candidate_pairs": len(candidates),
        }
    return result, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path, help="Bonn survey/samples.csv")
    parser.add_argument("--gateways", required=True, type=Path, help="Bonn survey/gateways.csv")
    parser.add_argument("--output", type=Path, default=Path("results/paper_experiment"))
    parser.add_argument("--split", choices=["random", "packet", "spatial_500m"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--models", nargs="+", choices=["rf", "xgboost"],
                        default=["rf", "xgboost"])
    parser.add_argument("--limit", type=int, help="Development smoke run; not for paper results")
    args = parser.parse_args()
    df, source = load_measurements(args.samples, args.gateways)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).sort_index()
        source["development_sample_rows"] = len(df)
    result, predictions = run_experiment(df, args.split, args.seed, args.jobs, args.models)
    result["source"] = source
    result["versions"] = {
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "xgboost": __import__("xgboost").__version__,
        "numpy": np.__version__,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    predictions.to_csv(args.output / "test_predictions.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
