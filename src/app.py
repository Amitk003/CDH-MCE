import os
import sys
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ingest import load_all_data, fill_missing_dates
from features import build_features, build_aggregate_features, merge_features
from predict import run_inference
from roas import compute_roas_range


st.set_page_config(
    page_title="CDH-MCE Forecast",
    layout="wide",
)


def load_forecast(data_dir, model_path, budget_multiplier=1.0):
    output_path = "output/ui_predictions.csv"
    run_inference(data_dir, model_path, output_path, budget_multiplier=budget_multiplier)
    return pd.read_csv(output_path)


def get_llm_insights(forecast_df, groq_client, model_name="llama-3.3-70b-versatile"):
    agg = forecast_df[forecast_df["level"] == "aggregate"].copy()
    channels = forecast_df[forecast_df["level"] == "channel"].copy()
    risky = forecast_df[
        (forecast_df["level"] == "campaign")
        & (forecast_df["roas_lower"] < 1.0)
        & (forecast_df["revenue_lower"] > 0)
    ]

    prompt = f"""
You are a digital marketing analyst. Given the following forecast data, provide a short business summary.

Aggregate forecasts:
{agg.to_string(index=False)}

Channel-level forecasts (30 day):
{channels[channels['horizon']==30].to_string(index=False)}

Campaigns where ROAS lower bound is below 1.0 (risk of loss):
{risky.to_string(index=False)}

Please provide:
1. A one-paragraph summary of the overall outlook.
2. Key risks to highlight.
3. One actionable recommendation.
"""

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate AI insights: {e}"


def main():
    st.title("CDH-MCE Revenue & ROAS Forecaster")
    st.markdown("---")

    with st.sidebar:
        st.header("Setup")

        data_source = st.radio(
            "Data source",
            ["Use data/ folder", "Upload CSV files"],
            index=0,
        )

        data_dir = "data"
        if data_source == "Upload CSV files":
            uploaded_files = st.file_uploader(
                "Upload CSV files",
                type=["csv"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                upload_dir = Path("output/uploaded_data")
                upload_dir.mkdir(parents=True, exist_ok=True)
                for f in uploaded_files:
                    with open(upload_dir / f.name, "wb") as fp:
                        fp.write(f.getbuffer())
                data_dir = str(upload_dir)

        model_path = st.text_input(
            "Model path",
            value="./pickle/model.pkl",
        )

        run_button = st.button("Run Forecast", type="primary")

        st.markdown("---")
        st.header("Budget Simulation")

        enable_budget = st.checkbox("Adjust budgets", value=False)
        budget_multiplier = st.slider(
            "Budget multiplier",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            disabled=not enable_budget,
        )

        st.markdown("---")
        st.header("AI Insights")

        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
        )

        model_name = st.selectbox(
            "LLM Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0,
        )

    if run_button:
        with st.spinner("Running forecast..."):
            try:
                pdf = load_forecast(
                    data_dir, model_path,
                    budget_multiplier=budget_multiplier if enable_budget else 1.0,
                )
                st.session_state["forecast_df"] = pdf
                st.session_state["forecast_ready"] = True
                st.success("Forecast complete!")
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                return

    if st.session_state.get("forecast_ready"):
        pdf = st.session_state["forecast_df"]

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Summary", "By Channel", "By Campaign Type", "All Campaigns", "AI Insights"]
        )

        with tab1:
            st.subheader("Aggregate Forecast")
            agg = pdf[pdf["level"] == "aggregate"].copy()

            cols = st.columns(3)
            for i, h in enumerate([30, 60, 90]):
                row = agg[agg["horizon"] == h]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                with cols[i]:
                    st.metric(
                        label=f"{h}-Day Revenue Range",
                        value=f"${r['revenue_lower']:,.0f} - ${r['revenue_upper']:,.0f}",
                    )
                    st.metric(
                        label=f"{h}-Day ROAS Range",
                        value=f"{r['roas_lower']:.2f} - {r['roas_upper']:.2f}",
                    )

            fig = go.Figure()
            for h in [30, 60, 90]:
                row = agg[agg["horizon"] == h]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                fig.add_trace(go.Bar(
                    name=f"{h}-Day",
                    x=["Revenue Lower", "Revenue Upper"],
                    y=[r["revenue_lower"], r["revenue_upper"]],
                ))
            fig.update_layout(
                title="Revenue Forecast by Horizon",
                yaxis_title="Revenue",
                barmode="group",
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Channel-Level Forecast")
            ch = pdf[pdf["level"] == "channel"].copy()

            horizon_filter = st.selectbox(
                "Select horizon",
                [30, 60, 90],
                key="ch_horizon",
            )
            ch_subset = ch[ch["horizon"] == horizon_filter]

            fig_rev = go.Figure()
            for _, row in ch_subset.iterrows():
                fig_rev.add_trace(go.Bar(
                    name=row["group"],
                    x=["Revenue Lower", "Revenue Upper"],
                    y=[row["revenue_lower"], row["revenue_upper"]],
                ))
            fig_rev.update_layout(
                title=f"Revenue by Channel ({horizon_filter}-Day)",
                yaxis_title="Revenue",
                barmode="group",
            )
            st.plotly_chart(fig_rev, use_container_width=True)

            st.dataframe(
                ch_subset[["group", "revenue_lower", "revenue_upper", "roas_lower", "roas_upper"]]
                .round(2)
                .rename(columns={
                    "group": "Channel",
                    "revenue_lower": "Revenue Lower",
                    "revenue_upper": "Revenue Upper",
                    "roas_lower": "ROAS Lower",
                    "roas_upper": "ROAS Upper",
                }),
                use_container_width=True,
            )

        with tab3:
            st.subheader("Campaign Type Level Forecast")
            ct = pdf[pdf["level"] == "campaign_type"].copy()

            horizon_filter2 = st.selectbox(
                "Select horizon",
                [30, 60, 90],
                key="ct_horizon",
            )
            ct_subset = ct[ct["horizon"] == horizon_filter2]

            fig_ct = px.bar(
                ct_subset,
                x="group",
                y=["revenue_lower", "revenue_upper"],
                barmode="group",
                title=f"Revenue by Campaign Type ({horizon_filter2}-Day)",
                labels={"group": "Campaign Type", "value": "Revenue"},
            )
            st.plotly_chart(fig_ct, use_container_width=True)

            st.dataframe(
                ct_subset[["group", "revenue_lower", "revenue_upper", "roas_lower", "roas_upper"]]
                .round(2)
                .rename(columns={
                    "group": "Campaign Type",
                    "revenue_lower": "Revenue Lower",
                    "revenue_upper": "Revenue Upper",
                    "roas_lower": "ROAS Lower",
                    "roas_upper": "ROAS Upper",
                }),
                use_container_width=True,
            )

        with tab4:
            st.subheader("All Campaign Forecasts")
            camp = pdf[pdf["level"] == "campaign"].copy()

            horizon_filter3 = st.selectbox(
                "Select horizon",
                [30, 60, 90],
                key="camp_horizon",
            )
            camp_subset = camp[camp["horizon"] == horizon_filter3]

            search = st.text_input("Search campaign name", "")
            if search:
                camp_subset = camp_subset[
                    camp_subset["group"].str.contains(search, case=False, na=False)
                ]

            st.dataframe(
                camp_subset[["group", "revenue_lower", "revenue_upper", "roas_lower", "roas_upper"]]
                .round(2)
                .rename(columns={
                    "group": "Campaign",
                    "revenue_lower": "Revenue Lower",
                    "revenue_upper": "Revenue Upper",
                    "roas_lower": "ROAS Lower",
                    "roas_upper": "ROAS Upper",
                }),
                use_container_width=True,
                height=400,
            )

            risky = camp_subset[
                (camp_subset["roas_lower"] < 1.0)
                & (camp_subset["revenue_lower"] > 0)
            ]
            if len(risky) > 0:
                st.warning(
                    f"{len(risky)} campaign(s) have ROAS lower bound below 1.0 "
                    "meaning they may not break even."
                )
                st.dataframe(
                    risky[["group", "revenue_lower", "roas_lower"]]
                    .round(2)
                    .rename(columns={
                        "group": "Campaign",
                        "revenue_lower": "Revenue Lower",
                        "roas_lower": "ROAS Lower",
                    }),
                    use_container_width=True,
                )

        with tab5:
            st.subheader("AI-Powered Insights")

            if not api_key:
                st.info(
                    "Enter your Groq API key in the sidebar to get AI-powered insights."
                )
            else:
                from groq import Groq

                client = Groq(api_key=api_key)

                if st.button("Generate Insights", type="primary"):
                    with st.spinner("Getting AI insights..."):
                        insights = get_llm_insights(pdf, client, model_name)
                    st.markdown(insights)

                st.markdown("---")
                st.markdown("#### Ask a custom question")
                user_question = st.text_area(
                    "Ask a question about the forecast data",
                    placeholder="e.g., Which channels are performing best?",
                )
                if user_question:
                    agg_str = pdf[pdf["level"] == "aggregate"].to_string(index=False)
                    question_prompt = f"""
Given this forecast data:
{agg_str}

Answer this question: {user_question}

Keep your answer brief and focused on the data.
"""
                    with st.spinner("Thinking..."):
                        try:
                            response = client.chat.completions.create(
                                messages=[{"role": "user", "content": question_prompt}],
                                model=model_name,
                                temperature=0.3,
                                max_tokens=300,
                            )
                            st.markdown(response.choices[0].message.content)
                        except Exception as e:
                            st.error(f"Error: {e}")

    else:
        st.info("Click 'Run Forecast' in the sidebar to start.")


if __name__ == "__main__":
    main()
