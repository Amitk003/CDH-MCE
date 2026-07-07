import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from ingest import load_all_data, fill_missing_dates
from features import build_features, build_aggregate_features, merge_features
from targets import build_targets, HORIZONS
from utils import set_seeds


QUANTILE_LOWER = 0.1
QUANTILE_UPPER = 0.9
CONFORMAL_COVERAGE = 0.9


TRAIN_PARAMS = {
    "objective": "quantile",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}


def train_quantile_model(X, y, alpha, params=None):
    merged_params = dict(TRAIN_PARAMS)
    if params:
        merged_params.update(params)
    merged_params["alpha"] = alpha
    model = lgb.LGBMRegressor(**merged_params)
    model.fit(X, y)
    return model


def run_training(data_dir, model_out, params=None):
    set_seeds(42)

    print("Loading data...")
    df = load_all_data(data_dir)
    df = fill_missing_dates(df)
    print(f"  Raw + filled: {df.shape}")

    print("Building features...")
    cf = build_features(df)
    af = build_aggregate_features(df)
    merged, feature_cols = merge_features(cf, af)
    print(f"  Features: {merged.shape}, {len(feature_cols)} columns")

    print("Building targets...")
    tdf = build_targets(df)
    merged = merged.merge(
        tdf[["campaign_name", "date"]
            + [c for c in tdf.columns if "target" in c]],
        on=["campaign_name", "date"],
        how="left",
    )

    target_cols = [c for c in merged.columns if "target" in c]
    print(f"  After target merge: {merged.shape}")

    all_dates = sorted(merged["date"].unique())
    split_idx = int(len(all_dates) * 0.8)
    train_dates = set(all_dates[:split_idx])
    cal_dates = set(all_dates[split_idx:])

    base_train = merged[merged["date"].isin(train_dates)].copy()
    base_cal = merged[merged["date"].isin(cal_dates)].copy()
    print(f"  Base train rows: {base_train.shape}, Cal rows: {base_cal.shape}")

    models = {}
    for h in HORIZONS:
        for metric in ["revenue", "spend"]:
            target = f"{metric}_target_{h}d"

            train_df = base_train.dropna(subset=[target])
            y_train = train_df[target].values
            X_train = train_df[feature_cols]

            print(f"  Training {metric} {h}d ({train_df.shape[0]} rows, q={QUANTILE_LOWER})...")
            models[f"{metric}_{h}d_lower"] = train_quantile_model(
                X_train, y_train, QUANTILE_LOWER, params
            )

            print(f"  Training {metric} {h}d ({train_df.shape[0]} rows, q={QUANTILE_UPPER})...")
            models[f"{metric}_{h}d_upper"] = train_quantile_model(
                X_train, y_train, QUANTILE_UPPER, params
            )

    print("Running conformal calibration...")
    q_crit = {}
    for h in HORIZONS:
        for metric in ["revenue", "spend"]:
            key = f"{metric}_{h}d"
            cal_df = base_cal.dropna(subset=[f"{metric}_target_{h}d"])
            X_cal = cal_df[feature_cols]
            y_cal = cal_df
            lower_key = f"{key}_lower"
            upper_key = f"{key}_upper"
            model_lower = models[lower_key]
            model_upper = models[upper_key]

            pred_lower = model_lower.predict(X_cal)
            pred_upper = model_upper.predict(X_cal)

            y_true = y_cal[f"{metric}_target_{h}d"].values
            scores = np.maximum(pred_lower - y_true, y_true - pred_upper)
            scores = np.maximum(scores, 0)
            q_crit[key] = float(np.percentile(scores, CONFORMAL_COVERAGE * 100))

    for k, v in q_crit.items():
        print(f"  q_crit[{k}] = {v:.4f}")

    model_package = {
        "models": models,
        "feature_cols": feature_cols,
        "q_crit": q_crit,
        "horizons": HORIZONS,
        "quantile_lower": QUANTILE_LOWER,
        "quantile_upper": QUANTILE_UPPER,
    }

    model_out_path = Path(model_out)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_out_path, "wb") as f:
        pickle.dump(model_package, f, protocol=4)
    print(f"Model saved to {model_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-out", default="./pickle/model.pkl")
    args = parser.parse_args()

    run_training(args.data_dir, args.model_out)
