"""Download, clean and aggregate the UCI Online Retail dataset into RFM features.

    python -m src.data    # builds data/processed/rfm.csv and cleaning_stats.json

Each raw row is one product line on an invoice. We clean the transactions,
then aggregate them into ONE row per customer:

  * Recency   = days between the customer's last purchase and the day after
                the last date in the dataset (the "snapshot" date)
  * Frequency = number of distinct invoices (orders)
  * Monetary  = total amount spent (quantity x unit price, £)
"""
import hashlib
import json
import urllib.request
import zipfile

import pandas as pd

from src.config import CLEANING_PATH, NON_PRODUCT_CODES, RAW_SHA256, RAW_URL, RAW_ZIP, RFM_PATH


def download_raw() -> None:
    if RAW_ZIP.exists():
        return
    RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
    tmp = RAW_ZIP.with_suffix(".part")
    print("Downloading Online Retail dataset (23 MB) ...")
    urllib.request.urlretrieve(RAW_URL, tmp)
    if hashlib.sha256(tmp.read_bytes()).hexdigest() != RAW_SHA256:
        tmp.unlink()
        raise RuntimeError("Downloaded file does not match the expected checksum.")
    tmp.replace(RAW_ZIP)


def load_transactions() -> pd.DataFrame:
    download_raw()
    with zipfile.ZipFile(RAW_ZIP) as z, z.open("Online Retail.xlsx") as f:
        return pd.read_excel(f, dtype={"InvoiceNo": str, "StockCode": str})


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply each cleaning rule in turn and record how many rows it removed."""
    steps = [{"step": "Raw transaction lines", "rows": len(df), "removed": 0}]

    def apply(mask, label):
        nonlocal df
        before = len(df)
        df = df[mask]
        steps.append({"step": label, "rows": len(df), "removed": before - len(df)})

    apply(df["CustomerID"].notna(), "Drop lines without a CustomerID (cannot be linked to a customer)")
    apply(~df["InvoiceNo"].str.startswith("C"), "Drop cancellations (invoice number starts with 'C')")
    apply((df["Quantity"] > 0) & (df["UnitPrice"] > 0), "Drop non-positive quantity or price")
    apply(~df["StockCode"].str.upper().isin(NON_PRODUCT_CODES),
          "Drop non-product codes (postage, fees, discounts, adjustments)")
    apply(~df.duplicated(), "Drop exact duplicate lines")

    df = df.assign(CustomerID=df["CustomerID"].astype(int), Revenue=df["Quantity"] * df["UnitPrice"])
    return df, steps


def build_rfm(tx: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    snapshot = tx["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = tx.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        frequency=("InvoiceNo", "nunique"),
        monetary=("Revenue", "sum"),
        country=("Country", lambda s: s.mode().iat[0]),
    )
    rfm["recency"] = (snapshot - rfm["last_purchase"].dt.normalize()).dt.days
    rfm = rfm.drop(columns="last_purchase").reset_index()
    return rfm[["CustomerID", "recency", "frequency", "monetary", "country"]], str(snapshot.date())


def build_and_save() -> tuple[pd.DataFrame, dict]:
    raw = load_transactions()
    tx, steps = clean_transactions(raw)
    rfm, snapshot = build_rfm(tx)
    stats = {
        "cleaning_steps": steps,
        "date_range": [str(tx["InvoiceDate"].min()), str(tx["InvoiceDate"].max())],
        "snapshot_date": snapshot,
        "n_customers": int(len(rfm)),
        "n_invoices": int(tx["InvoiceNo"].nunique()),
        "total_revenue": float(tx["Revenue"].sum()),
        "uk_customer_share": float((rfm["country"] == "United Kingdom").mean()),
    }
    RFM_PATH.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(RFM_PATH, index=False)
    CLEANING_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return rfm, stats


def load_rfm() -> pd.DataFrame:
    """Load the customer-level RFM table, building it if necessary."""
    if not RFM_PATH.exists():
        build_and_save()
    return pd.read_csv(RFM_PATH)


def load_cleaning_stats() -> dict:
    if not CLEANING_PATH.exists():
        build_and_save()
    return json.loads(CLEANING_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    table, info = build_and_save()
    for s in info["cleaning_steps"]:
        print(f"{s['rows']:>8,}  (-{s['removed']:,})  {s['step']}")
    print({k: v for k, v in info.items() if k != "cleaning_steps"})
    print(table.describe().round(1))
