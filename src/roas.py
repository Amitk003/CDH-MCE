import numpy as np
import pandas as pd


def compute_roas_range(revenue_lower, revenue_upper, spend_lower, spend_upper):
    """
    Compute blended ROAS range using cross-interval division.
    ROAS = Revenue / Spend.
    Lower bound uses worst case (lowest revenue / highest spend).
    Upper bound uses best case (highest revenue / lowest spend).
    """
    if spend_lower < 1e-9:
        return 0.0, 0.0
    roas_lower = revenue_lower / max(spend_upper, 1e-9)
    roas_upper = revenue_upper / max(spend_lower, 1e-9)
    return float(roas_lower), float(roas_upper)


def build_predictions(forecasts, horizons):
    """
    Build the final predictions DataFrame from raw forecast dicts.

    forecasts is a list of dicts with keys:
      horizon, level, group, revenue_lower, revenue_upper,
      spend_lower, spend_upper
    """
    rows = []
    for fc in forecasts:
        h = fc["horizon"]
        rl = fc["revenue_lower"]
        ru = fc["revenue_upper"]
        sl = fc["spend_lower"]
        su = fc["spend_upper"]

        roas_lower, roas_upper = compute_roas_range(rl, ru, sl, su)

        rows.append({
            "horizon": h,
            "level": fc["level"],
            "group": fc["group"],
            "revenue_lower": round(rl, 2),
            "revenue_upper": round(ru, 2),
            "roas_lower": round(roas_lower, 4),
            "roas_upper": round(roas_upper, 4),
        })

    return pd.DataFrame(rows)
