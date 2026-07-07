import pandas as pd
import numpy as np


LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 30]


def _compute_rolling(group, col, windows, suffix="sum"):
    for w in windows:
        group[f"{col}_{w}d_{suffix}"] = (
            group[col].shift(1).rolling(w, min_periods=1).sum().fillna(0)
        )
    return group


def _compute_lags(group, col, days):
    for d in days:
        group[f"{col}_lag_{d}"] = group[col].shift(d)
    return group


def _adstock(series, decay=0.5):
    result = np.zeros_like(series, dtype=float)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = series[i] + decay * result[i - 1]
    return result


def _compute_metrics(group):
    group["cpc"] = (group["spend"] / group["clicks"].replace(0, np.nan)).fillna(0)
    group["cpm"] = (group["spend"] / group["impressions"].replace(0, np.nan) * 1000).fillna(0)
    group["ctr"] = (group["clicks"] / group["impressions"].replace(0, np.nan) * 100).fillna(0)
    group["conversion_rate"] = (group["conversions"] / group["clicks"].replace(0, np.nan)).fillna(0)
    group["roas"] = (group["revenue"] / group["spend"].replace(0, np.nan)).fillna(0)
    group["budget_util"] = (group["spend"] / group["daily_budget"].replace(0, np.nan)).fillna(0)
    return group


def _compute_rolling_metrics(group, windows):
    for w in windows:
        for col in ["cpc", "cpm", "ctr", "conversion_rate", "roas", "budget_util"]:
            if col in group.columns:
                group[f"{col}_{w}d_avg"] = (
                    group[col].shift(1).rolling(w, min_periods=1).mean().fillna(0)
                )
    return group


def build_features(df):
    df = df.copy()
    df.sort_values(["campaign_name", "date"], inplace=True)

    # compute derived metrics per campaign
    groups = []
    for name, group in df.groupby("campaign_name", sort=False):
        group = group.copy()
        group = _compute_metrics(group)

        # adstock on spend
        group["spend_adstock"] = _adstock(group["spend"].values, decay=0.5)

        # lags
        for col in ["revenue", "spend", "conversions", "spend_adstock"]:
            if col in group.columns:
                group = _compute_lags(group, col, LAG_DAYS)

        # rolling sums
        for col in ["revenue", "spend", "conversions"]:
            if col in group.columns:
                group = _compute_rolling(group, col, ROLLING_WINDOWS, suffix="sum")

        # rolling means on metrics
        group = _compute_rolling_metrics(group, ROLLING_WINDOWS)

        groups.append(group)

    df = pd.concat(groups, ignore_index=True)

    # temporal features
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["dom_sin"] = np.sin(2 * np.pi * df["dom"] / 31)
    df["dom_cos"] = np.cos(2 * np.pi * df["dom"] / 31)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # channel encoding
    df["is_google"] = (df["channel"] == "google_ads").astype(int)
    df["is_meta"] = (df["channel"] == "meta_ads").astype(int)
    df["is_bing"] = (df["channel"] == "bing_ads").astype(int)

    # campaign type one-hot
    ct_dummies = pd.get_dummies(df["campaign_type"], prefix="ct")
    df = pd.concat([df, ct_dummies], axis=1)

    return df


def build_aggregate_features(df):
    agg = df.groupby("date", as_index=False).agg({
        "spend": "sum",
        "revenue": "sum",
        "clicks": "sum",
        "impressions": "sum",
        "conversions": "sum",
    })
    agg.columns = ["date", "agg_spend", "agg_revenue", "agg_clicks",
                   "agg_impressions", "agg_conversions"]

    agg["agg_cpc"] = (agg["agg_spend"] / agg["agg_clicks"].replace(0, np.nan)).fillna(0)
    agg["agg_cpm"] = (agg["agg_spend"] / agg["agg_impressions"].replace(0, np.nan) * 1000).fillna(0)
    agg["agg_ctr"] = (agg["agg_clicks"] / agg["agg_impressions"].replace(0, np.nan) * 100).fillna(0)
    agg["agg_roas"] = (agg["agg_revenue"] / agg["agg_spend"].replace(0, np.nan)).fillna(0)

    agg["agg_spend_lag_1"] = agg["agg_spend"].shift(1)
    agg["agg_revenue_lag_1"] = agg["agg_revenue"].shift(1)

    for w in [7, 14, 30]:
        agg[f"agg_revenue_{w}d_sum"] = (
            agg["agg_revenue"].shift(1).rolling(w, min_periods=1).sum().fillna(0)
        )
        agg[f"agg_spend_{w}d_sum"] = (
            agg["agg_spend"].shift(1).rolling(w, min_periods=1).sum().fillna(0)
        )

    return agg


NON_FEATURE_COLS = [
    "date", "campaign_name", "channel", "campaign_type",
    "spend", "revenue", "clicks", "impressions", "conversions",
    "daily_budget"
]


def merge_features(campaign_features, aggregate_features,
                   training_feature_cols=None):
    merged = campaign_features.merge(
        aggregate_features, on="date", how="left"
    )

    if training_feature_cols is not None:
        for col in training_feature_cols:
            if col not in merged.columns:
                merged[col] = 0
        merged = merged[
            [c for c in merged.columns if c not in training_feature_cols]
            + training_feature_cols
        ]
        merged.dropna(subset=training_feature_cols, inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged, training_feature_cols

    feature_cols = [c for c in merged.columns
                    if c not in NON_FEATURE_COLS]
    merged.dropna(subset=feature_cols, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged, feature_cols
