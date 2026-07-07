import pandas as pd
import numpy as np


HORIZONS = [30, 60, 90]


def build_targets(df, horizons=None):
    if horizons is None:
        horizons = HORIZONS
    df = df.copy()
    df.sort_values(["campaign_name", "date"], inplace=True)

    groups = []
    for name, group in df.groupby("campaign_name", sort=False):
        group = group.copy()
        for h in horizons:
            # revenue target: forward sum of h days
            group[f"revenue_target_{h}d"] = (
                group["revenue"]
                .shift(-h)
                .rolling(h, min_periods=1)
                .sum()
                .shift(-h + 1)
            )
            # spend target
            group[f"spend_target_{h}d"] = (
                group["spend"]
                .shift(-h)
                .rolling(h, min_periods=1)
                .sum()
                .shift(-h + 1)
            )
            # null out incomplete windows at the end
            group.loc[
                group[f"revenue_target_{h}d"].isna(),
                f"revenue_target_{h}d"
            ] = np.nan
            group.loc[
                group[f"spend_target_{h}d"].isna(),
                f"spend_target_{h}d"
            ] = np.nan
        groups.append(group)

    result = pd.concat(groups, ignore_index=True)
    return result


def build_aggregate_targets(df, horizons=None):
    if horizons is None:
        horizons = HORIZONS
    agg = df.groupby("date", as_index=False).agg({
        "spend": "sum",
        "revenue": "sum",
    })
    agg.sort_values("date", inplace=True)
    agg.columns = ["date", "agg_spend", "agg_revenue"]

    for h in horizons:
        agg[f"agg_revenue_target_{h}d"] = (
            agg["agg_revenue"]
            .shift(-h)
            .rolling(h, min_periods=1)
            .sum()
            .shift(-h + 1)
        )
        agg[f"agg_spend_target_{h}d"] = (
            agg["agg_spend"]
            .shift(-h)
            .rolling(h, min_periods=1)
            .sum()
            .shift(-h + 1)
        )
    return agg
