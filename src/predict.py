import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ingest import load_all_data, fill_missing_dates, validate_campaigns
from features import build_features, build_aggregate_features, merge_features
from targets import HORIZONS
from roas import build_predictions
from utils import set_seeds


def generate_llm_summary(result_df, api_key, model="llama3-70b-8192"):
    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        agg_rows = result_df[result_df["level"] == "aggregate"]
        summary_lines = []
        for _, r in agg_rows.iterrows():
            summary_lines.append(
                f"- {r['horizon']}d: revenue [{r['revenue_lower']:.0f}, {r['revenue_upper']:.0f}], "
                f"ROAS [{r['roas_lower']:.2f}, {r['roas_upper']:.2f}]"
            )
        agg_table = "\n".join(summary_lines)

        prompt = (
            "You are a marketing analytics expert. Given these forecast ranges "
            f"(aggregate level):\n{agg_table}\n\n"
            "Provide a concise 3-4 sentence analysis: key trends, risk assessment, "
            "and actionable recommendation."
        )

        chat = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=300,
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM summary unavailable: {e}"


def run_inference(data_dir, model_path, output_path, budget_multiplier=1.0,
                  llm_api_key=None, llm_model="llama3-70b-8192"):
    set_seeds(42)

    print("Loading data...")
    df = load_all_data(data_dir)
    df = fill_missing_dates(df)
    validate_campaigns(df)

    if budget_multiplier != 1.0:
        df["daily_budget"] = df["daily_budget"] * budget_multiplier
        print(f"  Budget scaled by {budget_multiplier}x")

    print(f"  Rows: {df.shape}")

    print("Loading model...")
    with open(model_path, "rb") as f:
        pkg = pickle.load(f)

    models = pkg["models"]
    feature_cols = pkg["feature_cols"]
    q_crit = pkg["q_crit"]
    horizons = pkg["horizons"]
    quantile_lower = pkg["quantile_lower"]
    quantile_upper = pkg["quantile_upper"]

    print("Building features...")
    cf = build_features(df)
    af = build_aggregate_features(df)
    merged, _ = merge_features(cf, af, training_feature_cols=feature_cols)
    print(f"  Rows with features: {merged.shape}")

    last_date = merged["date"].max()
    print(f"  Last feature date: {last_date}")

    last_rows = merged[merged["date"] == last_date].copy()
    print(f"  Campaigns at last date: {last_rows['campaign_name'].nunique()}")

    X_last = last_rows[feature_cols]
    camp_names = last_rows["campaign_name"].values
    channels = last_rows["channel"].values
    camp_types = last_rows["campaign_type"].values

    forecasts = []
    for h in horizons:
        rev_lower = models[f"revenue_{h}d_lower"].predict(X_last)
        rev_upper = models[f"revenue_{h}d_upper"].predict(X_last)
        spend_lower = models[f"spend_{h}d_lower"].predict(X_last)
        spend_upper = models[f"spend_{h}d_upper"].predict(X_last)

        q = q_crit[f"revenue_{h}d"]
        rev_lower -= q
        rev_upper += q
        qs = q_crit[f"spend_{h}d"]
        spend_lower -= qs
        spend_upper += qs

        rev_lower = np.maximum(rev_lower, 0)
        rev_upper = np.maximum(rev_upper, 0)
        spend_lower = np.maximum(spend_lower, 0)
        spend_upper = np.maximum(spend_upper, 0)

        for i in range(len(camp_names)):
            forecasts.append({
                "horizon": h,
                "level": "campaign",
                "campaign_name": camp_names[i],
                "channel": channels[i],
                "campaign_type": camp_types[i],
                "revenue_lower": rev_lower[i],
                "revenue_upper": rev_upper[i],
                "spend_lower": spend_lower[i],
                "spend_upper": spend_upper[i],
            })

    fc_df = pd.DataFrame(forecasts)
    print(f"  Campaign forecasts: {fc_df.shape}")

    # build aggregate-level forecasts
    agg_forecasts = []
    for h in horizons:
        subset = fc_df[fc_df["horizon"] == h]
        total_rev_lower = subset["revenue_lower"].sum()
        total_rev_upper = subset["revenue_upper"].sum()
        total_spend_lower = subset["spend_lower"].sum()
        total_spend_upper = subset["spend_upper"].sum()

        agg_forecasts.append({
            "horizon": h,
            "level": "aggregate",
            "campaign_name": "all",
            "channel": "all",
            "campaign_type": "all",
            "revenue_lower": total_rev_lower,
            "revenue_upper": total_rev_upper,
            "spend_lower": total_spend_lower,
            "spend_upper": total_spend_upper,
        })

        for ch in subset["channel"].unique():
            ch_subset = subset[subset["channel"] == ch]
            agg_forecasts.append({
                "horizon": h,
                "level": "channel",
                "campaign_name": "all",
                "channel": ch,
                "campaign_type": "all",
                "revenue_lower": ch_subset["revenue_lower"].sum(),
                "revenue_upper": ch_subset["revenue_upper"].sum(),
                "spend_lower": ch_subset["spend_lower"].sum(),
                "spend_upper": ch_subset["spend_upper"].sum(),
            })

        for ct in subset["campaign_type"].unique():
            ct_subset = subset[subset["campaign_type"] == ct]
            agg_forecasts.append({
                "horizon": h,
                "level": "campaign_type",
                "campaign_name": "all",
                "channel": "all",
                "campaign_type": ct,
                "revenue_lower": ct_subset["revenue_lower"].sum(),
                "revenue_upper": ct_subset["revenue_upper"].sum(),
                "spend_lower": ct_subset["spend_lower"].sum(),
                "spend_upper": ct_subset["spend_upper"].sum(),
            })

    all_forecasts = agg_forecasts + forecasts
    result_df = build_predictions(all_forecasts, horizons)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"Predictions written to {output_path}")
    print(f"  Total rows: {result_df.shape[0]}")
    print(f"  Columns: {list(result_df.columns)}")

    if llm_api_key:
        print("Generating LLM summary...")
        summary = generate_llm_summary(result_df, llm_api_key, llm_model)
        summary_path = out_path.with_suffix(".summary.txt")
        summary_path.write_text(summary, encoding="utf-8")
        print(f"LLM summary written to {summary_path}")
        print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model", default="./pickle/model.pkl")
    parser.add_argument("--output", default="./output/predictions.csv")
    parser.add_argument("--budget-multiplier", type=float, default=1.0)
    parser.add_argument("--llm-api-key", default=None,
                        help="Groq API key for optional LLM insights (omitted by default)")
    parser.add_argument("--llm-model", default="llama3-70b-8192")
    args = parser.parse_args()

    run_inference(
        args.data_dir, args.model, args.output,
        budget_multiplier=args.budget_multiplier,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
    )
