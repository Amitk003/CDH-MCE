import numpy as np
import pandas as pd


MAX_ROAS = 999.99


def compute_roas_range(revenue_lower, revenue_upper, spend_lower, spend_upper):
    if spend_upper < 1e-9:
        return 0.0, 0.0

    roas_lower = revenue_lower / spend_upper

    if spend_lower < 1e-9:
        roas_upper = min(MAX_ROAS, revenue_upper / 1e-9)
    else:
        roas_upper = min(MAX_ROAS, revenue_upper / spend_lower)

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
