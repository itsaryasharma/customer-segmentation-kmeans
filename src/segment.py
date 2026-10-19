"""K-Means pipeline, segment ordering, naming and profiling.

K-Means numbers its clusters arbitrarily (cluster 0 is not "best"). To make
results stable and readable we:
  1. rank clusters by a value score computed from their centroids, and
  2. name each cluster from its centroid's position relative to the average
     customer (rules below), so names are derived from the data, not invented.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import N_INIT, RANDOM_STATE, RFM_FEATURES

# Suggested actions are hypotheses for a marketing team to test (e.g. with an
# A/B test). The data shows past behaviour; it does not show these actions work.
ACTIONS = {
    "Champions": "Reward and retain: loyalty perks, early access to new products, ask for reviews or "
                 "referrals. Avoid unnecessary discounts; these customers already buy at full price.",
    "Loyal customers": "Keep engaged: loyalty programme, personalised recommendations. These are the most "
                       "likely candidates to become Champions.",
    "Regular mid-value": "Grow: cross-sell related products, bundles and volume offers to raise order "
                         "frequency or basket size.",
    "New or occasional": "Develop: welcome/onboarding e-mails, product recommendations and a second-purchase "
                         "incentive to turn a first order into a habit.",
    "At-risk valuable": "Win back: personalised reminders based on past purchases, a time-limited offer, "
                        "or a short survey to learn why they stopped buying.",
    "Lapsed low-value": "Low-cost reactivation only (e.g. an occasional newsletter). Avoid expensive "
                        "campaigns until they show renewed interest.",
}
DESCRIPTIONS = {
    "Champions": "Bought very recently, order often and spend the most.",
    "Loyal customers": "Recent repeat buyers with above-average spend.",
    "Regular mid-value": "Buy at an average pace with moderate to good spend.",
    "New or occasional": "Bought fairly recently, but few and small orders so far.",
    "At-risk valuable": "Used to order often or spend well, but have not bought for a while.",
    "Lapsed low-value": "Few, small orders and no recent purchase.",
}
NAMING_RULES = [  # (recency rule, value rule, name), shown in the app and docs
    ("z_R ≤ −0.5 (recent)", "value ≥ 1", "Champions"),
    ("z_R ≤ −0.5 (recent)", "0 ≤ value < 1", "Loyal customers"),
    ("z_R < 0.5", "value < 0", "New or occasional"),
    ("−0.5 < z_R < 0.5 (average)", "value ≥ 0", "Regular mid-value"),
    ("z_R ≥ 0.5 (not recent)", "value ≥ 0", "At-risk valuable"),
    ("z_R ≥ 0.5 (not recent)", "value < 0", "Lapsed low-value"),
]


def make_pipeline(k: int, seed: int = RANDOM_STATE) -> Pipeline:
    """log1p -> standardise -> K-Means.

    * log1p: RFM values are extremely right-skewed (monetary ranges from £4 to
      £279,000). Without it a few huge customers dominate the distances.
    * StandardScaler: puts days, order counts and pounds on the same scale, so
      no feature dominates just because of its units.
    """
    return Pipeline([
        ("log", FunctionTransformer(np.log1p, inverse_func=np.expm1, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=N_INIT, random_state=seed)),
    ])


def scaled_centroids(model: Pipeline) -> np.ndarray:
    """Centroids in standardised log space (rows = clusters, cols = R, F, M)."""
    return model.named_steps["kmeans"].cluster_centers_


def centroids_original_units(model: Pipeline) -> np.ndarray:
    """Centroids converted back to days, orders and pounds."""
    c = model.named_steps["scale"].inverse_transform(scaled_centroids(model))
    return np.expm1(c)


def value_rank(model: Pipeline) -> dict[int, int]:
    """Map raw cluster id -> rank (0 = most valuable).

    Value score of a centroid = -recency + frequency + monetary (all standardised),
    i.e. recent, frequent, high-spending customers rank first.
    """
    c = scaled_centroids(model)
    score = -c[:, 0] + c[:, 1] + c[:, 2]
    order = np.argsort(-score)
    return {int(raw): rank for rank, raw in enumerate(order)}


def base_name(z: np.ndarray) -> str:
    """Name a centroid from its standardised (R, F, M) position (see NAMING_RULES).

    z is measured in standard deviations from the average customer (in log
    space). Negative z_R means MORE recent than average. value = (z_F + z_M) / 2.
    A threshold of 0.5 standard deviations separates "clearly different from
    average" from "about average".
    """
    zr, zf, zm = z
    value = (zf + zm) / 2
    if zr <= -0.5 and value >= 1:
        return "Champions"
    if zr <= -0.5 and value >= 0:
        return "Loyal customers"
    if zr < 0.5 and value < 0:
        return "New or occasional"
    if zr < 0.5:
        return "Regular mid-value"
    if value >= 0:
        return "At-risk valuable"
    return "Lapsed low-value"


def segment_names(model: Pipeline) -> list[str]:
    """Names in rank order; duplicates get a tier suffix (e.g. 'Champions (tier 2)')."""
    rank = value_rank(model)
    by_rank = sorted(rank, key=rank.get)
    names, seen = [], {}
    for raw in by_rank:
        name = base_name(scaled_centroids(model)[raw])
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name} (tier {seen[name]})")
    return names


def assign_segments(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Segment rank (0 = most valuable) for each row of X."""
    rank = value_rank(model)
    return np.array([rank[int(c)] for c in model.predict(X[RFM_FEATURES])])


def action_for(name: str) -> str:
    return ACTIONS[name.split(" (tier")[0]]


def description_for(name: str) -> str:
    return DESCRIPTIONS[name.split(" (tier")[0]]


def profile(rfm: pd.DataFrame, segments: np.ndarray, names: list[str]) -> list[dict]:
    """Per-segment statistics computed from the actual customers."""
    total_rev = rfm["monetary"].sum()
    out = []
    for rank, name in enumerate(names):
        g = rfm[segments == rank]
        out.append({
            "rank": rank,
            "name": name,
            "description": description_for(name),
            "action": action_for(name),
            "customers": int(len(g)),
            "customer_share": float(len(g) / len(rfm)),
            "revenue": float(g["monetary"].sum()),
            "revenue_share": float(g["monetary"].sum() / total_rev),
            "median": {f: float(g[f].median()) for f in RFM_FEATURES},
            "mean": {f: float(g[f].mean()) for f in RFM_FEATURES},
        })
    return out
