# Development Log: Customer Segmentation

All work took place on **19 October 2026**. Environment: Windows 11, Python 3.13.1; packages pinned in
`requirements.txt` (scikit-learn 1.9.1, Streamlit 1.66.0, pandas 3.0.6, openpyxl 3.1.5, …).

## 1. Dataset choice
- **Considered:** "Mall Customers" (200 rows; Gender, Age, Income, Spending Score), the most common
  dataset for this task. **Rejected:** too small, of unclear and probably synthetic origin, and not
  real behaviour, so any "insight" from it would not be genuine.
- **Chosen:** UCI Online Retail (541,909 real transaction lines, CC BY 4.0). Confirmed the licence and the
  DOI (10.24432/C5BW33) on the UCI page. The dataset's own original study (Chen, Sain & Guo, 2012) uses
  RFM + K-Means, which supports the method.

## 2. Profiling and cleaning
- Reading the Excel file takes about 90 seconds, so the pipeline caches a processed customer table (`rfm.csv`).
- Found: 135,080 lines without CustomerID; 9,288 cancellation lines; 10,624 non-positive quantities;
  2,517 non-positive prices; 5,268 duplicate lines; non-product stock codes (POST, DOT, M, C2, D, S, BANK
  CHARGES, AMAZONFEE, CRUK, PADS, B).
- Implemented `clean_transactions` so that every rule records how many rows it removed (applied in
  sequence, so later counts are smaller than the raw counts above): 541,909 → 391,150 lines →
  **4,334 customers**.
- RFM summary: monetary median £663, max £279,138; frequency median 2, max 206. Extreme skew, so a log
  transform is needed before scaling.

## 3. First clustering run
- Pipeline `log1p → StandardScaler → KMeans(n_init=20)`; evaluated k = 2…10.
- Initial rule: highest silhouette in k = 3…6, which selected **k = 3** (0.3382; k = 4: 0.3361). Stability
  ARI 0.994.
- **Problem: naming.** The first naming rules only checked above/below average, so the middle cluster
  (median 30 days, 3 orders, £955) was named "Champions (tier 2)". That was misleading.

## 4. Investigation and decisions
- Compared the profiles for k = 3, 4 and 5:
  - k = 3: Champions / middle group / lapsed. The middle group mixes recent low-volume buyers with established regulars.
  - k = 4: Champions (9 d, 10 orders, £3,683) / regular mid-value (53 d, 4, £1,359) / new-or-occasional
    (19 d, 2, £440) / lapsed (182 d, 1, £300). These are four groups needing **different** actions.
- **Decision (a judgement call, documented as such):** the silhouette difference between k = 3 and k = 4
  (0.002) is negligible, so the rule was revised to *"the largest k in 3–6 within 0.01 of the best
  silhouette"*, which selects k = 4. This change was made **after** seeing the profiles; the docs and
  the app say so openly. In unsupervised learning there is no held-out test set that this could
  contaminate, but the choice remains subjective.
- **Naming rewritten** with explicit thresholds on the standardised centroid (±0.5 std for recency;
  value = (z_F + z_M)/2) and six possible names. Checked that k = 2…6 give sensible, unique names (tier
  suffixes only appear at k ≥ 7).

## 5. Final run
```
 k   inertia silhouette     DB       CH
 2    6469.8     0.4328  0.890   4373.8
 3    4850.1     0.3382  1.041   3639.8
 4    3918.8     0.3361  1.018   3345.4
 5    3262.4     0.3182  0.983   3231.0
 6    2835.9     0.3134  1.014   3103.0
Chosen k: 4 | stability: mean ARI 0.970, min 0.947 (10 seeds)
  [0] Champions          n=699  (16.1%)  revenue 64.4%  median R=9d   F=10 M=£3683
  [1] Regular mid-value  n=1159 (26.7%)  revenue 24.0%  median R=53d  F=4  M=£1359
  [2] New or occasional  n=878  (20.3%)  revenue 5.2%   median R=19d  F=2  M=£440
  [3] Lapsed low-value   n=1598 (36.9%)  revenue 6.5%   median R=182d F=1  M=£300
```

## 6. App and tests
- Five-tab app: Segments (incl. CSV export), Score a customer, Choosing k (with a live k explorer), How it works, Data.
- The hero banner originally hard-coded "4,334 customers"; it now reads the number from metrics.
- 25 tests written; all passed on the first run (see TESTING.md). Headless server check: health `ok`, HTTP 200.
- Installed `requirements.txt` into a brand-new virtual environment and ran the full suite there: **25 passed**
  (after moving the environment to a shorter path to avoid `WinError 206`; see the README's Windows note).

## 7. Not done / future work
- Product-category or return-rate features; GMM/DBSCAN comparison; monthly segment tracking.
- Not tested on mobile screens.

## 8. Refinement phase (19 October 2026)
- **Independent re-verification from the raw Excel file**, with the cleaning and RFM re-implemented without `src/data.py`: 391,150 lines, 4,334 customers, 18,402 invoices and £8,737,227.64, all identical; the RFM table is identical to `rfm.csv`. Silhouette recomputed (0.336101) equals the stored value. Scoring a customer agrees with `model.predict` for 300 sampled customers. Every CSV export contains exactly that segment's customers.
- **Stability:** the stored check (10 random seeds, ARI 0.970) only tests initialisation. Added a stronger check, 20 refits on random 80% subsamples (ARI 0.923, min 0.872), stored by `train.py` and shown in the app with an explanation of both.
- **Retraining note:** after the addition, centroid values differed by at most 4.5 × 10⁻¹³ (floating-point summation order in parallel K-Means); assignments and all reported figures are identical.
- **Visual and explanation updates:** an R/F/M explainer row, segment icons (Streamlit's bundled Material Symbols, no image files), and a table comparing each segment's typical customer with the overall median (× multiples). Native light/dark themes, checked with screenshots in both modes.
- **Docs:** added `docs/MODEL_CARD.md`; the long-form study notes are no longer part of the repository; README quick start added.
