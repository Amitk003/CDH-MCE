# Architecture Overview

## System Pipeline

```
CSV Files (Google Ads, Meta Ads, Bing Ads)
    |
    v
ingest.py  -->  Reads CSVs, standardizes columns, classifies campaign types
    |
    v
features.py  -->  Creates feature columns (lags, rolling windows, adstock, time features)
    |
    v
targets.py  -->  Computes 30/60/90 day forward sums for revenue and spend
    |
    v
train.py  -->  Trains LightGBM quantile models and runs conformal calibration
    |
    v
pickle/model.pkl  -->  Saved model file (12 models + calibration data)
    |
    v
predict.py  -->  Loads model, builds features for latest data, runs inference
    |
    v
roas.py  -->  Computes ROAS ranges from revenue and spend forecasts
    |
    v
output/predictions.csv  -->  Final forecast file (CSV format)
```

## Frontend Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI Framework | Streamlit | Web-based dashboard for running forecasts and viewing results |
| Charts | Plotly | Interactive bar charts for channel and campaign type breakdowns |
| AI Integration | Groq API (Llama 3.3 70B) | Generates business summaries and answers questions |

The Streamlit app has 5 tabs:

1. **Summary**: Shows aggregate revenue and ROAS ranges for 30/60/90 days with a bar chart
2. **By Channel**: Breaks down forecasts by ad platform (Google, Meta, Bing)
3. **By Campaign Type**: Breaks down by campaign type (Search, PMax, Social, etc.)
4. **All Campaigns**: Lists every campaign with its forecast, searchable, with risk flagging
5. **AI Insights**: Groq-powered analysis and custom Q&A about the forecast

## Backend Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.11 | All code is Python |
| ML Model | LightGBM 4.6.0 | Gradient boosting for quantile regression |
| Data Processing | Pandas 2.2.2, NumPy 1.26.4 | Data manipulation and feature engineering |
| Serialization | Pickle | Model saving and loading |
| CLI Entry Point | Bash (run.sh) | Automated pipeline execution |

## Forecasting Pipeline

### Training phase (run offline, not in the automated test)

1. Load CSV files from the data/ folder
2. Standardize schemas across Google Ads, Meta Ads, and Bing Ads
3. Fill missing dates for each campaign
4. Create 88 feature columns per campaign per day
5. Compute target values (forward sums of revenue and spend for 30/60/90 days)
6. Split data chronologically: 80% training, 20% calibration
7. Train 12 LightGBM models (lower and upper quantile for each metric and horizon)
8. Run conformal calibration on the held-out calibration set
9. Save all models and calibration data to a pickle file

### Inference phase (run during testing)

1. Load CSV files (the test data replaces the development data)
2. Build features using the same pipeline as training
3. Load the saved model pickle
4. For each campaign, predict lower and upper bounds for revenue and spend
5. Apply the conformal calibration adjustment (q_crit values)
6. Aggregate predictions to channel, campaign type, and total levels
7. Compute ROAS ranges using cross-interval division
8. Write predictions to output CSV

## LLM Integration Workflow

1. The forecast is generated using the ML models (no LLM involved at this stage)
2. Forecast data is formatted as text (aggregate numbers, channel breakdown, risky campaigns)
3. A prompt is created asking the LLM for:
   - A one-paragraph business summary
   - Key risks
   - One actionable recommendation
4. The prompt is sent to the Groq API using a Llama model
5. The response is displayed in the Streamlit UI
6. If no API key is available, a message is shown instead of crashing

The LLM is only used in the Streamlit UI, not in the automated testing pipeline. The core forecasting pipeline (run.sh -> predict.py) does not make any network calls.
