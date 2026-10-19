# Model Card: Customer Segmentation (RFM + K-Means)

## Overview
| | |
|---|---|
| Task | Unsupervised clustering of customers by purchasing behaviour |
| Model | `Pipeline(FunctionTransformer(log1p), StandardScaler, KMeans(n_clusters=4, n_init=20, init="k-means++", random_state=42))` |
| Data | UCI Online Retail (CC BY 4.0): 541,909 transaction lines, Dec 2010 – Dec 2011, UK online giftware retailer |
| Features | Recency (days since last purchase, snapshot 2011-12-10), Frequency (distinct invoices), Monetary (total spend, £) |
| Code | `src/data.py`, `src/segment.py`, `src/train.py`, `src/predict.py` |
| Artifacts | `data/processed/rfm.csv`, `models/kmeans_model.joblib`, `models/metrics.json` |

## Data preparation
Lines without a CustomerID (135,080), cancellations (8,905), non-positive quantity or price (40),
non-product codes such as postage and fees (1,547) and exact duplicates (5,187) were removed →
391,150 lines → **4,334 customers**, 18,402 orders, £8,737,227.64. These figures were independently
re-computed from the raw file. Returns are excluded (gross RFM).

`log(1 + x)` handles extreme skew (spend from £3.75 to £279,138); standardisation gives each feature equal weight.

## Choosing k
| k | Inertia | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|---|---|---|---|---|
| 2 | 6469.8 | 0.433 | 0.890 | 4373.8 |
| 3 | 4850.1 | 0.338 | 1.041 | 3639.8 |
| **4** | 3918.8 | **0.336** | 1.018 | 3345.4 |
| 5 | 3262.4 | 0.318 | 0.983 | 3231.0 |
| 6 | 2835.9 | 0.313 | 1.014 | 3103.0 |

Rule: among k = 3–6, the largest k whose silhouette is within 0.01 of the best. k = 2 scores best on all
metrics but is too coarse for distinct actions. k = 3 and k = 4 are effectively tied, and k = 4 separates new
customers from regulars. **This is a practical modelling decision, not an objective optimum.**

**Stability** (adjusted Rand index vs the final model): 10 refits with new random seeds → 0.970 (min 0.947);
20 refits on random 80% samples of customers → 0.923 (min 0.872).

## Segments
| Segment | Customers | Revenue share | Median recency / orders / spend |
|---|---|---|---|
| Champions | 699 (16.1%) | 64.4% | 9 days / 10 / £3,683 |
| Regular mid-value | 1,159 (26.7%) | 24.0% | 53 days / 4 / £1,359 |
| New or occasional | 878 (20.3%) | 5.2% | 19 days / 2 / £440 |
| Lapsed low-value | 1,598 (36.9%) | 6.5% | 182 days / 1 / £300 |

Segments are ranked by a value score of the centroid (`−z_R + z_F + z_M`) and named by documented
threshold rules (`src/segment.py::NAMING_RULES`).

## Interpretation and limitations
- A silhouette of ≈ 0.34 indicates **overlapping** groups: customer behaviour is a continuum, and boundary customers could fit a neighbouring segment.
- Suggested marketing actions are **hypotheses** to test (e.g. A/B tests). Their effect has not been measured.
- RFM ignores what customers buy, margins and returns; one retailer, one year, mostly UK wholesale customers.
- 25% of lines lacked a CustomerID and could not be used.
- K-Means assumes compact, similarly sized clusters and always returns k groups.
