import csv
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

CHANNEL_TYPE_MAPPING = {
    "SEARCH": "Search",
    "PERFORMANCE_MAX": "PMax",
    "DISPLAY": "Display",
    "VIDEO": "Video",
    "DEMAND_GEN": "Demand_Gen",
    "SHOPPING": "Shopping",
}


def classify_campaign_type(campaign_name, channel_type=None):
    if pd.isna(campaign_name):
        return "Other"
    name = str(campaign_name)
    for pattern, label in CAMPAIGN_TYPE_PATTERNS:
        if re.search(pattern, name):
            return label
    if channel_type and pd.notna(channel_type):
        ct = str(channel_type).upper()
        return CHANNEL_TYPE_MAPPING.get(ct, "Other")
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
    df["campaign_type"] = [
        classify_campaign_type(n, t)
        for n, t in zip(df["campaign_name"], df["campaign_type_raw"])
    ]
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
    df["campaign_type"] = [
        classify_campaign_type(n, t)
        for n, t in zip(df["campaign_name"], df["campaign_type_raw"])
    ]
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
    df["campaign_type"] = [
        classify_campaign_type(n, None)
        for n in df["campaign_name"]
    ]
    df["conversions"] = np.nan
    return df


# Ordered list of (keyword tuples, reader function) so we check
# more specific patterns first and cover aliases (microsoft, ms, etc.)
READER_RULES = [
    (["google", "ga4", "adwords"], read_google),
    (["meta", "facebook", "fb"], read_meta),
    (["bing", "microsoft", "ms", "microsoft_ads"], read_bing),
]


# Column signatures per reader for fallback detection when keywords don't match
READER_SIGNATURES = {
    read_google: {"segments_date", "metrics_cost_micros", "campaign_advertising_channel_type"},
    read_meta: {"date_start", "cpc", "cpm", "daily_budget"},
    read_bing: {"TimePeriod", "CampaignId", "CampaignType"},
}


def _match_reader(fname, csv_columns=None):
    fname = fname.lower()
    for keywords, reader in READER_RULES:
        for kw in keywords:
            if kw in fname:
                return reader

    if csv_columns is not None:
        cols_lower = {c.lower() for c in csv_columns}
        for reader, sig in READER_SIGNATURES.items():
            if sig.issubset(cols_lower) or sig.issubset(set(csv_columns)):
                return reader

    return None


def load_all_data(data_dir):
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    all_dfs = []
    for fpath in csv_files:
        reader = _match_reader(fpath.name)
        if reader is None:
            with open(fpath, encoding="utf-8-sig") as f:
                sample = f.readline(65536)
            head = list(csv.reader([sample]))[0] if sample.strip() else []
            reader = _match_reader(fpath.name, csv_columns=head)
            if reader is None:
                print(f"  WARNING: No reader matched for {fpath.name}, skipping")
                continue
        df = reader(fpath)
        all_dfs.append(df)

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


def validate_campaigns(df):
    issues = []

    for name, group in df.groupby("campaign_name", sort=False):
        group = group.sort_values("date")

        neg_spend = group[group["spend"] < -0.001]
        for _, r in neg_spend.iterrows():
            issues.append(
                f"  [{name}] Negative spend ({r['spend']}) on {r['date']}"
            )

        neg_rev = group[group["revenue"] < -0.001]
        for _, r in neg_rev.iterrows():
            issues.append(
                f"  [{name}] Negative revenue ({r['revenue']}) on {r['date']}"
            )

        dups = group[group.duplicated(subset=["date"], keep=False)]
        if len(dups) > 0:
            dup_dates = dups["date"].unique()
            issues.append(
                f"  [{name}] Duplicate entries for dates: {list(dup_dates[:5])}"
            )

        total_rev = group["revenue"].sum()
        total_spend = group["spend"].sum()
        if total_spend < 0.01 and total_rev < 0.01:
            issues.append(
                f"  [{name}] Campaign has no spend or revenue (all zeros)"
            )

    if issues:
        print("Campaign validation issues found:")
        for msg in issues:
            print(msg)
    else:
        print("Campaign validation passed: no issues found")

    return issues


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

    global_max_date = df["date"].max()
    result_dfs = []
    for name, group in df.groupby("campaign_name", sort=False):
        campaign_start = group["date"].min()
        campaign_dates = pd.date_range(
            start=campaign_start, end=global_max_date, freq="D"
        )
        group = group.set_index("date").reindex(campaign_dates)
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
