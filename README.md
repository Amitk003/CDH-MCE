# CDH-MCE

A forecasting tool for ecommerce marketing data. It predicts future revenue and ROAS using historical ad data from Google Ads, Meta Ads, and Bing Ads.

## What it does

- Reads campaign data from CSV files
- Generates features like rolling averages, lags, and adstock transformations
- Trains LightGBM models to predict revenue and spend over 30/60/90 day windows
- Uses conformal prediction to give probabilistic ranges (lower and upper bounds)
- Computes blended ROAS ranges from revenue and spend forecasts
- Provides a Streamlit UI for running forecasts and getting AI-powered insights

## How to run

### Setup

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python src/train.py --data-dir ./data --model-out ./pickle/model.pkl
```

### Run prediction

```bash
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

### Launch the UI

```bash
streamlit run src/app.py
```

## Project structure

```
src/
  ingest.py        - Reads CSV files and standardizes them
  features.py      - Creates features for the model
  targets.py       - Builds 30/60/90 day target values
  train.py         - Trains LightGBM models with conformal calibration
  predict.py       - Runs inference and produces output
  roas.py          - Computes blended ROAS ranges
  llm_insights.py  - AI-powered analysis using Groq
  app.py           - Streamlit user interface
  utils.py         - Shared helper functions

data/              - Input CSV files
pickle/            - Trained model file
output/            - Forecast results
docs/              - Documentation
```

## Requirements

- Python 3.10 or higher
- See requirements.txt for all dependencies
