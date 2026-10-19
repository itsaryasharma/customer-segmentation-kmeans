# Testing: Customer Segmentation

## How to run
```bash
pytest        # from the project folder; 25 tests, ~20 seconds
```
The tests use the committed `data/processed/rfm.csv` and do not need the 23 MB raw download.

## What is tested

| Area | Test(s) |
|---|---|
| **Cleaning rules** | A synthetic 6-row table: each rule (missing customer, cancellation, zero price, postage code, duplicate) removes exactly one row; revenue = quantity × price |
| **RFM definitions** | Hand-computed example: recency 5 days, frequency 2 invoices, monetary £21; snapshot date = day after the last transaction |
| RFM table validity | Unique customers; no missing values; recency ≥ 1, frequency ≥ 1, monetary > 0 |
| **Metric honesty** | Segment sizes recomputed from the saved model equal the stored sizes; silhouette recomputed equals the stored value; revenue shares sum to 1 |
| Ranking | Segment 0 has the highest median spend |
| k rule | Unit test with known scores: k = 2 excluded; a near-tie picks the larger k; otherwise the best |
| Naming rules | All six names produced by the documented thresholds; names unique for the saved model |
| Scoring | A big recent spender → Champions; one-off, long ago → Lapsed low-value; the nearest centre agrees with the label |
| Validation (6 cases) | Recency 0, frequency 0, negative spend, fractional orders, NaN, non-numeric: all rejected |
| Range warnings | None for typical values; one for 500 orders |
| App (`AppTest`) | 5 tabs render; scoring presets give the expected segments; the k-explorer slider re-clusters without errors and shows the "exploratory" note |

## Results (19 October 2026)
```
.........................                                                [100%]
25 passed in 18.95s
```
Run with `-W error::DeprecationWarning -W error::FutureWarning`. `python -W error -m src.data` and
`python -W error -m src.train` complete without warnings. Headless server: health `ok`, HTTP 200.

## Not covered automatically
Visual layout (including the 3-D chart) in a browser; mobile screens; the raw download step (tested
manually on 19 Oct 2026, checksum matched).

## Screenshots to capture for the report
1. Segments tab: KPI row + segment cards.
2. Customer share vs revenue share chart, and the segment fingerprints.
3. 3-D scatter (rotate to a clear angle).
4. Score a customer: "Big recent spender" with the distance chart.
5. Choosing k: the three metric charts, plus the explorer at k = 3 for comparison.
6. Data tab: cleaning steps + raw vs log histogram.

## Refinement-phase verification (19 October 2026)
Command: `python -m pytest -q -W error::DeprecationWarning -W error::FutureWarning` → **25 passed**.

Additional scripted checks: cleaning and RFM re-implemented independently from the raw Excel file (4,334
customers, 18,402 invoices, £8,737,227.64; RFM table identical); silhouette recomputed (equal); scoring
agrees with `model.predict` for 300 customers; each CSV export contains exactly its segment's customers;
subsample stability added (ARI 0.923); screenshots of the Segments and Choosing-k tabs in both themes.

**Fresh-clone check:** only the files Git would publish were copied, a new virtual environment was created with
`py -3.13 -m venv`, `pip install -r requirements.txt` succeeded, `python -m pytest` gave 25 passed, and
`python -m streamlit run app.py` started (health `ok`, HTTP 200).
