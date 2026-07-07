# Technical Documentation

## Forecasting Methodology

### What the model does

The system predicts future revenue and ROAS (Return on Ad Spend) for 30, 60, and 90 day periods. It uses historical data from Google Ads, Meta Ads, and Bing Ads campaigns.

Instead of predicting day-by-day and adding up (which makes errors grow over time), the model is trained to predict the total sum directly. This is called "direct-horizon forecasting".

### How predictions are made

For each campaign, the model looks at the most recent day of data and predicts:

- How much revenue the campaign will generate in the next 30/60/90 days
- How much money the campaign will spend in the next 30/60/90 days

These predictions are then:

1. Added up across campaigns to get channel-level and total forecasts
2. Used to calculate ROAS ranges (revenue divided by spend)

### Probabilistic ranges (not single numbers)

The model outputs a lower bound and an upper bound for each prediction, rather than a single number. This gives a range that the real value is expected to fall inside.

The ranges are built using two techniques:

1. **Quantile regression**: Two models are trained for each prediction. One predicts the 10th percentile (lower bound) and one predicts the 90th percentile (upper bound). Together they form an 80% confidence range.

2. **Conformal prediction**: The raw ranges from the quantile models are adjusted using a calibration step. This step looks at how far off the model was on held-out data and expands the ranges to make sure they cover the real value at the target confidence level. Conformal prediction works for any type of data without assuming a normal distribution.

---

## Model Selection

### Why LightGBM

We chose LightGBM over other options for these reasons:

| Model | Problem |
|-------|---------|
| ARIMA | Assumes linear patterns, struggles with marketing data that has sudden changes |
| LSTM / RNN | Needs a lot of data and tuning, slow to train, hard to debug |
| Prophet | Designed for daily forecasts with seasonality, not for direct-horizon targets |
| **LightGBM** | Fast, handles non-linear patterns, works well with sparse data, supports quantile regression natively |

LightGBM is a gradient boosting model that builds many decision trees. Each tree tries to fix the errors of the previous ones. It is:

- Fast to train (minutes instead of hours)
- Works on CPU (no GPU needed)
- Handles missing values automatically
- Supports quantile regression (needed for the probabilistic ranges)

### Training setup

- 12 models total: for each metric (revenue, spend) and each horizon (30, 60, 90), we train a lower-bound model and an upper-bound model
- 80% of data used for training, 20% for conformal calibration
- Chronological split (no future data leaking into training)

---

## Data Preprocessing

### Reading the CSV files

The system reads three different CSV formats and standardizes them:

| Source | Date column | Revenue column | Spend column | Notes |
|--------|-------------|----------------|--------------|-------|
| Google Ads | segments_date | metrics_conversions_value | metrics_cost_micros / 1,000,000 | Spend is in micros (divide by 1M) |
| Meta Ads | date_start | conversion | spend | Direct values |
| Bing Ads | TimePeriod | Revenue | Spend | Direct values |

### Campaign type classification

Campaign names are checked against patterns to figure out the campaign type:

- Names containing "Search" are classified as Search
- Names containing "PMax" or "Performance Max" are classified as PMax
- Names containing "Prospecting" are classified as Social_Prospecting
- Names containing "Remarketing" are classified as Social_Remarketing
- And so on for other types

If the name pattern does not match, the raw channel type from the data is used as a backup.

### Missing data handling

- Campaigns are reindexed to have every day from their start date to the end date
- Missing spend and revenue values are filled with 0
- Missing campaign metadata is forward-filled from the previous available day

### Feature engineering

For each campaign, the following features are created:

1. **Lags**: Revenue, spend, and conversions from 1, 7, 14, and 28 days ago
2. **Rolling sums**: Total revenue, spend, and conversions over the last 7, 14, and 30 days
3. **Adstock**: A smoothed version of spend that accounts for the delayed effect of advertising (a geometric decay with rate 0.5)
4. **Derived metrics**: CPC, CPM, CTR, conversion rate, ROAS, and budget utilization
5. **Rolling averages of derived metrics**: 7, 14, and 30 day averages
6. **Time features**: Day of week, day of month, and month encoded as sine/cosine waves
7. **Channel indicators**: Whether the campaign is from Google, Meta, or Bing
8. **Campaign type indicators**: One-hot encoding of the campaign type
9. **Aggregate features**: Same features computed across all campaigns combined

---

## Assumptions

1. **Existing attribution is correct**: The system uses the conversion and revenue data as given. It does not try to re-attribute conversions across channels.

2. **Future will look somewhat like the past**: The model learns patterns from historical data and assumes similar patterns will continue. Sudden major changes (like a new competitor or platform policy change) will not be captured.

3. **Campaigns continue to run**: The forecast assumes campaigns will keep running with similar budgets and settings.

4. **Daily data is available**: The system expects daily-level data for each campaign.

5. **Spend is a driver of revenue**: The model uses past spend levels to predict future revenue. If spend patterns change drastically, the predictions will be less reliable.

---

## Limitations

1. **Cold start for new campaigns**: Campaigns with less than 30 days of history have limited feature data. The adaptive windowing (falling back from 30 to 14 to 7 day windows) helps but predictions for new campaigns are less reliable.

2. **No external factors**: The model does not account for holidays, promotions, competitor actions, or economic changes. These can cause actual results to fall outside the predicted ranges.

3. **ROAS ranges can be wide**: When the lower bound of predicted spend is very small, the upper bound of ROAS can become very large (dividing by a tiny number). The system caps ROAS at 999.99 to keep it readable.

4. **Conformal prediction coverage**: The conformal calibration targets 90% coverage on the calibration set. On completely new data with different patterns, actual coverage may be lower.

5. **Campaign-level aggregation**: The aggregate forecasts are the sum of campaign-level forecasts. Correlations between campaigns (one campaign's success affecting another) are not modeled.

---

## AI Integration Strategy

### What the LLM is used for

The system uses Groq with a Llama model to generate:

1. **Business summaries**: A plain-language overview of the forecast results
2. **Risk identification**: Which campaigns have ROAS lower bound below 1.0 (meaning they may lose money)
3. **Actionable recommendations**: Suggestions based on the forecast data

### How it works

1. After the forecast is generated, the forecast data is formatted as text
2. A prompt is sent to the LLM asking for:
   - A one-paragraph summary
   - Key risks to highlight
   - One actionable recommendation
3. The LLM response is displayed in the Streamlit UI

### Custom questions

Users can also ask their own questions about the forecast data. The question is sent to the LLM along with the aggregate forecast numbers.

### Graceful fallback

If no API key is provided, or if the API call fails, the system shows a message explaining that the key is needed. It does not crash or stop working.
