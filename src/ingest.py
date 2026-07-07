import pandas as pd
import numpy as np
from pathlib import Path
import re


CAMPAIGN_TYPE_PATTERNS = [
    (r"(?i)prospecting.*adv", "Social_Prospecting"),
    (r"(?i)prospecting.*dpa", "Social_Prospecting"),
    (r"(?i)prospecting.*brand", "Social_Prospecting"),
    (r"(?i)prospecting", "Social_Prospecting"),
    (r"(?i)remarketing.*dpa", "Social_Remarketing"),
    (r"(?i)remarketing.*brand", "Social_Remarketing"),
    (r"(?i)remarketing", "Social_Remarketing"),
    (r"(?i)generic.*brand", "Social_Generic"),
    (r"(?i)generic", "Social_Generic"),
    (r"(?i)search.*tm", "Search"),
    (r"(?i)search.*ntm", "Search"),
    (r"(?i)search", "Search"),
    (r"(?i)shopping", "Shopping"),
    (r"(?i)pmax.*ntm", "PMax"),
    (r"(?i)pmax", "PMax"),
    (r"(?i)display", "Display"),
    (r"(?i)video.*ntm", "Video"),
    (r"(?i)video", "Video"),
    (r"(?i)demand.*gen", "Demand_Gen"),
]


def classify_campaign_type(campaign_name, channel_type=None):
    if pd.isna(campaign_name):
        return "Other"
    name = str(campaign_name)
    for pattern, label in CAMPAIGN_TYPE_PATTERNS:
        if re.search(pattern, name):
            return label
    if channel_type and pd.notna(channel_type):
        ct = str(channel_type).upper()
        mapping = {
            "SEARCH": "Search",
            "PERFORMANCE_MAX": "PMax",
            "DISPLAY": "Display",
            "VIDEO": "Video",
            "DEMAND_GEN": "Demand_Gen",
            "SHOPPING": "Shopping",
        }
        return mapping.get(ct, "Other")
    return "Other"


def read_bing(filepath):
    df = pd.read_csv(filepath)
    df.rename(columns={
        "TimePeriod": "date",
        "CampaignName": "campaign_name",
        "CampaignType": "campaign_type_raw",
        "Spend": "spend",
        "Revenue": "revenue",
        "Clicks": "clicks",
        "Impressions": "impressions",
        "Conversions": "conversions",
        "DailyBudget": "daily_budget",
    }, inplace=True)
    df["channel"] = "bing_ads"
    df["campaign_type"] = df["campaign_name"].apply(classify_campaign_type)
    return df


def read_google(filepath):
    df = pd.read_csv(filepath)
    df.rename(columns={
        "segments_date": "date",
        "campaign_name": "campaign_name",
        "campaign_advertising_channel_type": "campaign_type_raw",
        "metrics_clicks": "clicks",
        "metrics_conversions": "conversions",
        "metrics_impressions": "impressions",
        "metrics_video_views": "video_views",
        "metrics_conversions_value": "revenue",
        "campaign_budget_amount": "daily_budget",
    }, inplace=True)
    df["spend"] = df["metrics_cost_micros"] / 1_000_000
    df["channel"] = "google_ads"
    df["campaign_type"] = df.apply(
        lambda row: classify_campaign_type(
            row["campaign_name"], row.get("campaign_type_raw")
        ),
        axis=1,
    )
    return df


def read_meta(filepath):
    df = pd.read_csv(filepath)
    df.rename(columns={
        "date_start": "date",
        "campaign_name": "campaign_name",
        "spend": "spend",
        "clicks": "clicks",
        "impressions": "impressions",
        "conversion": "revenue",
        "daily_budget": "daily_budget",
    }, inplace=True)
    df["channel"] = "meta_ads"
    df["campaign_type"] = df["campaign_name"].apply(classify_campaign_type)
    df["conversions"] = np.nan
    return df


def load_all_data(data_dir):
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    readers = {
        "bing": read_bing,
        "google": read_google,
        "meta": read_meta,
    }

    all_dfs = []
    for fpath in csv_files:
        fname = fpath.name.lower()
        loaded = False
        for key, reader in readers.items():
            if key in fname:
                df = reader(fpath)
                all_dfs.append(df)
                loaded = True
                break
        if not loaded:
            print(f"Warning: Unknown CSV format, skipping {fpath.name}")

    if not all_dfs:
        raise ValueError("No matching CSV files found in data directory")

    df = pd.concat(all_dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    cols = [
        "date", "channel", "campaign_name", "campaign_type",
        "spend", "revenue", "clicks", "impressions", "conversions",
        "daily_budget"
    ]
    available = [c for c in cols if c in df.columns]
    df = df[available]

    df.sort_values(["campaign_name", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def fill_missing_dates(df):
    df = df.copy()
    df = df.groupby(
        ["campaign_name", "date"], as_index=False
    ).agg({
        "channel": "first",
        "campaign_type": "first",
        "spend": "sum",
        "revenue": "sum",
        "clicks": "sum",
        "impressions": "sum",
        "conversions": "sum",
        "daily_budget": "first",
    })
    all_dates = pd.date_range(
        start=df["date"].min(), end=df["date"].max(), freq="D"
    )
    result_dfs = []
    for name, group in df.groupby("campaign_name", sort=False):
        group = group.set_index("date").reindex(all_dates)
        group["campaign_name"] = name
        for col in ["channel", "campaign_type"]:
            if col in group.columns:
                group[col] = group[col].ffill().bfill()
        fill_cols = ["spend", "revenue", "clicks", "impressions", "conversions"]
        for c in fill_cols:
            if c in group.columns:
                group[c] = group[c].fillna(0)
        if "daily_budget" in group.columns:
            group["daily_budget"] = group["daily_budget"].ffill().fillna(0)
        result_dfs.append(group)
    result = pd.concat(result_dfs)
    result.index.name = "date"
    result.reset_index(inplace=True)
    return result
