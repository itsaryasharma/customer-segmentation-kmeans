"""Choose the number of clusters, fit K-Means and profile the segments.

    python -m src.train

1. For k = 2..10: fit the pipeline and record inertia (elbow method),
   silhouette score, Davies-Bouldin index and Calinski-Harabasz index.
2. Choose k: the highest silhouette score among k = 3..6 (see config for why
   this range). This is a judgement call, documented as such.
3. Fit the final model, rank and name the segments, profile them.
4. Stability check: refit with 10 different random seeds and measure how much
   the assignments agree (adjusted Rand index; 1.0 = identical).
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.config import (
    K_CANDIDATES,
    K_RANGE,
    LIMITS,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    RFM_FEATURES,
    STABILITY_SEEDS,
)
from src.data import load_cleaning_stats, load_rfm
from src.segment import (
    assign_segments,
    centroids_original_units,
    make_pipeline,
    profile,
    segment_names,
    value_rank,
)

SILHOUETTE_TOLERANCE = 0.01


def evaluate_k(X) -> list[dict]:
    results = []
    for k in K_RANGE:
        model = make_pipeline(k).fit(X)
        Z = model[:-1].transform(X)             # the space K-Means actually clusters in
        labels = model.named_steps["kmeans"].labels_
        results.append({
            "k": k,
            "inertia": float(model.named_steps["kmeans"].inertia_),
            "silhouette": float(silhouette_score(Z, labels)),
            "davies_bouldin": float(davies_bouldin_score(Z, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(Z, labels)),
        })
    return results


def choose_k(results: list[dict]) -> int:
    """Largest k in K_CANDIDATES whose silhouette is within the tolerance of the best.

    Near-equal silhouette scores mean near-equal cluster separation; in that
    case the extra segment gives more actionable detail at no measurable cost.
    """
    candidates = [r for r in results if r["k"] in K_CANDIDATES]
    best = max(r["silhouette"] for r in candidates)
    return max(r["k"] for r in candidates if r["silhouette"] >= best - SILHOUETTE_TOLERANCE)


def stability(X, k: int, reference_labels) -> dict:
    scores = []
    for seed in range(STABILITY_SEEDS):
        labels = make_pipeline(k, seed=seed + 1000).fit(X).named_steps["kmeans"].labels_
        scores.append(adjusted_rand_score(reference_labels, labels))
    return {"mean_ari": float(np.mean(scores)), "min_ari": float(np.min(scores)), "n_seeds": STABILITY_SEEDS}


def subsample_stability(X, k: int, reference_labels, n: int = 20, frac: float = 0.8) -> dict:
    """Stronger stability check: refit on random 80% subsamples of customers.

    Each refit labels ALL customers; the adjusted Rand index against the final
    model shows whether the segmentation depends on which customers happened
    to be in the data, not just on the random starting centres.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    for i in range(n):
        idx = rng.choice(len(X), size=int(frac * len(X)), replace=False)
        model = make_pipeline(k, seed=RANDOM_STATE + 100 + i).fit(X.iloc[idx])
        scores.append(adjusted_rand_score(reference_labels, model.predict(X)))
    return {"mean_ari": float(np.mean(scores)), "min_ari": float(np.min(scores)),
            "n_subsamples": n, "fraction": frac}


def train():
    rfm = load_rfm()
    X = rfm[RFM_FEATURES]
    k_results = evaluate_k(X)
    k = choose_k(k_results)

    model = make_pipeline(k).fit(X)
    names = segment_names(model)
    segments = assign_segments(model, X)
    rank = value_rank(model)
    centers = centroids_original_units(model)
    z_centers = model.named_steps["kmeans"].cluster_centers_
    by_rank = sorted(rank, key=rank.get)

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "data": load_cleaning_stats(),
        "k_evaluation": k_results,
        "chosen_k": k,
        "k_rule": (
            f"Among k = {min(K_CANDIDATES)}..{max(K_CANDIDATES)}, the largest k whose silhouette score is "
            f"within {SILHOUETTE_TOLERANCE} of the best. k = 2 has the highest silhouette overall, but it is "
            "excluded because a two-way split is too coarse for distinct marketing actions; more than 6 "
            "segments is hard to act on. Elbow, Davies-Bouldin and Calinski-Harabasz are reported as "
            "supporting evidence. Choosing k is a judgement call, not an objective optimum."
        ),
        "chosen": next(r for r in k_results if r["k"] == k),
        "stability": stability(X, k, model.named_steps["kmeans"].labels_),
        "subsample_stability": subsample_stability(X, k, model.named_steps["kmeans"].labels_),
        "segments": profile(rfm, segments, names),
        "centroids": [
            {"name": names[r], **{f: float(centers[raw][i]) for i, f in enumerate(RFM_FEATURES)},
             "z": [float(v) for v in z_centers[raw]]}
            for r, raw in enumerate(by_rank)
        ],
        "overall_median": {f: float(X[f].median()) for f in RFM_FEATURES},
        "ranges": {f: [float(X[f].min()), float(X[f].max())] for f in RFM_FEATURES},
        "limits": LIMITS,
        "random_state": RANDOM_STATE,
    }
    return model, metrics, rfm.assign(segment=segments)


def main() -> dict:
    model, metrics, _ = train()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"{'k':>2} {'inertia':>9} {'silhouette':>10} {'DB':>6} {'CH':>8}")
    for r in metrics["k_evaluation"]:
        print(f"{r['k']:>2} {r['inertia']:>9.1f} {r['silhouette']:>10.4f} {r['davies_bouldin']:>6.3f} "
              f"{r['calinski_harabasz']:>8.1f}")
    print("Chosen k:", metrics["chosen_k"], "| init stability:", metrics["stability"],
          "| subsample stability:", metrics["subsample_stability"])
    for s in metrics["segments"]:
        m = s["median"]
        print(f"  [{s['rank']}] {s['name']:<24} n={s['customers']:>5} ({s['customer_share']:.1%})  "
              f"revenue {s['revenue_share']:.1%}  median R={m['recency']:.0f}d F={m['frequency']:.0f} "
              f"M=£{m['monetary']:.0f}")
    for c in metrics["centroids"]:
        print("   centroid", c["name"], [round(v, 2) for v in c["z"]])
    return metrics


if __name__ == "__main__":
    main()
