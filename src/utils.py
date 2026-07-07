import numpy as np
import random
import os


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


METRIC_LABELS = {
    "revenue": "revenue",
    "spend": "spend",
}

HORIZON_LABELS = {30: "30d", 60: "60d", 90: "90d"}
