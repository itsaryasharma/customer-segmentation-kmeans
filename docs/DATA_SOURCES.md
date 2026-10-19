# Data Sources: Customer Segmentation

| | |
|---|---|
| Dataset | Online Retail |
| Repository | UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/352/online+retail |
| Download | https://archive.ics.uci.edu/static/public/352/online+retail.zip (contains `Online Retail.xlsx`) |
| Citation | Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33 |
| Original study | Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197–208 |
| Licence | **Creative Commons Attribution 4.0 International (CC BY 4.0)**: sharing and adaptation are allowed with attribution |
| SHA-256 (zip) | `f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b` (downloaded 19 Oct 2026; verified by `src/data.py`) |

## What is committed
- `data/raw/online_retail.zip` (23 MB): **not committed** (`.gitignore`); downloaded on demand.
- `data/processed/rfm.csv` (4,334 rows, 150 KB): **committed**. It is derived from CC BY 4.0 data, with
  attribution given here and in the app. CustomerIDs are the dataset's anonymous numeric IDs.
- `data/processed/cleaning_stats.json`: the counts below.

## Raw columns
`InvoiceNo` (a "C" prefix means cancellation), `StockCode`, `Description`, `Quantity`, `InvoiceDate`,
`UnitPrice` (£), `CustomerID`, `Country`.

## Cleaning steps (actual counts)

| Rows left | Removed | Step |
|---|---|---|
| 541,909 | — | Raw transaction lines |
| 406,829 | 135,080 | Drop lines without a CustomerID (cannot be linked to a customer) |
| 397,924 | 8,905 | Drop cancellations (invoice number starts with "C") |
| 397,884 | 40 | Drop non-positive quantity or price |
| 396,337 | 1,547 | Drop non-product codes: POST, DOT, C2, M, D, S, B, BANK CHARGES, AMAZONFEE, CRUK, PADS |
| 391,150 | 5,187 | Drop exact duplicate lines |

Result: **4,334 customers**, 18,402 invoices, £8,737,228 revenue; 90.4% of customers are in the UK.
Snapshot date for recency: **2011-12-10** (the day after the last transaction).
