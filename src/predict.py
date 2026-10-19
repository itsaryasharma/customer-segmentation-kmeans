"""Loading the model and scoring a single customer (no Streamlit code)."""
import json
import math

import joblib
import numpy as np
import pandas as pd

from src.config import LIMITS, METRICS_PATH, MODEL_PATH, RFM_FEATURES
from src.segment import segment_names, value_rank


def ensure_trained() -> None:
    if not (MODEL_PATH.exists() and METRICS_PATH.exists()):
        from src.train import main as train_main
        train_main()


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def load_metrics(path=METRICS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_customer(values: dict) -> list[str]:
    errors = []
    for f in RFM_FEATURES:
        v = values.get(f)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            errors.append(f"{f} must be a number.")
            continue
        lo, hi = LIMITS[f]
        if not lo <= v <= hi:
            errors.append(f"{f} must be between {lo:g} and {hi:g}.")
    if not errors and values["frequency"] != int(values["frequency"]):
        errors.append("frequency must be a whole number of orders.")
    return errors


def range_warnings(values: dict, ranges: dict) -> list[str]:
    out = []
    for f in RFM_FEATURES:
        lo, hi = ranges[f]
        if not lo <= values[f] <= hi:
            out.append(f"{f} = {values[f]:g} is outside the range seen in the data ({lo:g}–{hi:g}).")
    return out


def score_customer(model, values: dict) -> dict:
    """Assign a customer to a segment and report distances to every centroid."""
    errors = validate_customer(values)
    if errors:
        raise ValueError(" ".join(errors))
    X = pd.DataFrame([{f: float(values[f]) for f in RFM_FEATURES}])
    z = model[:-1].transform(X)[0]                      # log + standardise
    centers = model.named_steps["kmeans"].cluster_centers_
    rank = value_rank(model)
    names = segment_names(model)
    dist = np.linalg.norm(centers - z, axis=1)          # Euclidean distance, as K-Means uses
    raw = int(np.argmin(dist))
    return {
        "rank": rank[raw],
        "name": names[rank[raw]],
        "z": [float(v) for v in z],
        "distances": {names[rank[i]]: float(d) for i, d in enumerate(dist)},
    }
