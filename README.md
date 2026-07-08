# CDH-MCE

A forecasting tool for ecommerce marketing data. It predicts future revenue and ROAS (Return on Ad Spend) using historical ad data from Google Ads, Meta Ads, and Bing Ads.

## What it does

- Reads campaign data from CSV files (Google Ads, Meta Ads, Bing Ads)
- Standardizes the different data formats into a single structure
- Creates features like rolling averages, lags, and adstock transformations
- Trains LightGBM models to predict revenue and spend over 30/60/90 day windows
- Uses conformal prediction to give probabilistic ranges (lower and upper bounds)
- Computes blended ROAS ranges from revenue and spend forecasts
- Provides a Streamlit UI for running forecasts and getting AI-powered insights
- Supports budget simulation (what-if analysis with different spend levels)

## How to run

### Setup

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python src/train.py --data-dir ./data --model-out ./pickle/model.pkl
```

### Run prediction (command line)

```bash
chmod +x run.sh          # make sure it is executable (run once)
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

Optional: adjust budgets with --budget-multiplier:

```bash
python src/predict.py --data-dir ./data --model ./pickle/model.pkl --output ./output/predictions.csv --budget-multiplier 1.5
```

Optional: generate AI-powered summary of the forecast (requires Groq API key):

```bash
python src/predict.py --data-dir ./data --model ./pickle/model.pkl --output ./output/predictions.csv --llm-api-key gsk_...
```

> **Note:** The `--llm-api-key` flag is optional and disabled by default.
> The automated pipeline (`run.sh`) does not make network calls.

### Launch the UI

```bash
streamlit run src/app.py
```

Then open your browser at the URL shown in the terminal (usually http://localhost:8501).

### Get AI insights

To use AI-powered insights in the Streamlit UI, get a free API key from https://console.groq.com and enter it in the sidebar.

## Project structure

```
src/
  ingest.py        - Reads CSV files and standardizes them
  features.py      - Creates features for the model
  targets.py       - Builds 30/60/90 day target values
  train.py         - Trains LightGBM models with conformal calibration
  predict.py       - Runs inference and produces output
  roas.py          - Computes blended ROAS ranges
  app.py           - Streamlit user interface
  utils.py         - Shared helper functions

data/              - Input CSV files
pickle/            - Trained model file
output/            - Forecast results
docs/              - Documentation
```

## Output format

The prediction output is a CSV file with these columns (same identifying columns as the training data, plus forecasted metrics):

| Column | Description |
|--------|-------------|
| horizon | Forecast window in days (30, 60, or 90) |
| level | Aggregation level (aggregate, channel, campaign_type, or campaign) |
| campaign_name | Campaign name ("all" at aggregate or channel/campaign_type levels) |
| channel | Channel name ("all" at aggregate or campaign_type levels) |
| campaign_type | Campaign type ("all" at aggregate or channel levels) |
| revenue_lower | Lower bound of revenue forecast |
| revenue_upper | Upper bound of revenue forecast |
| spend_lower | Lower bound of spend forecast |
| spend_upper | Upper bound of spend forecast |
| roas_lower | Lower bound of ROAS forecast (revenue / spend) |
| roas_upper | Upper bound of ROAS forecast |

## Requirements

- Python 3.11.9 (tested; any 3.10+ should work)
- See requirements.txt for all dependencies
