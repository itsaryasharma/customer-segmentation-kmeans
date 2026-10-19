"""Tests for cleaning, RFM features, clustering, naming and scoring."""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

from src.config import K_CANDIDATES, RFM_FEATURES
from src.data import build_rfm, clean_transactions, load_rfm
from src.predict import ensure_trained, load_metrics, load_model, range_warnings, score_customer, validate_customer
from src.segment import assign_segments, base_name, segment_names
from src.train import choose_k


@pytest.fixture(scope="module")
def model():
    ensure_trained()
    return load_model()


@pytest.fixture(scope="module")
def metrics():
    ensure_trained()
    return load_metrics()


def _tx(**overrides):
    row = dict(InvoiceNo="1001", StockCode="85123A", Description="X", Quantity=2,
               InvoiceDate=pd.Timestamp("2011-12-01 10:00"), UnitPrice=3.0, CustomerID=1.0, Country="United Kingdom")
    row.update(overrides)
    return row


def test_cleaning_rules_each_remove_the_right_rows():
    raw = pd.DataFrame([
        _tx(),                                   # kept
        _tx(CustomerID=np.nan),                  # no customer
        _tx(InvoiceNo="C1002", Quantity=-2),     # cancellation
        _tx(InvoiceNo="1003", UnitPrice=0.0),    # zero price
        _tx(InvoiceNo="1004", StockCode="POST"), # postage, not a product
        _tx(),                                   # exact duplicate of the first row
    ])
    clean, steps = clean_transactions(raw)
    assert len(clean) == 1
    assert [s["removed"] for s in steps[1:]] == [1, 1, 1, 1, 1]
    assert clean["Revenue"].iloc[0] == 6.0


def test_rfm_definitions():
    tx = pd.DataFrame([
        _tx(InvoiceNo="1", InvoiceDate=pd.Timestamp("2011-12-01"), Quantity=1, UnitPrice=10.0),
        _tx(InvoiceNo="1", InvoiceDate=pd.Timestamp("2011-12-01"), Quantity=2, UnitPrice=5.0, StockCode="X2"),
        _tx(InvoiceNo="2", InvoiceDate=pd.Timestamp("2011-12-05"), Quantity=1, UnitPrice=1.0),
        _tx(InvoiceNo="3", InvoiceDate=pd.Timestamp("2011-12-09"), CustomerID=2.0),
    ])
    tx = tx.assign(CustomerID=tx["CustomerID"].astype(int), Revenue=tx["Quantity"] * tx["UnitPrice"])
    rfm, snapshot = build_rfm(tx)
    c1 = rfm.set_index("CustomerID").loc[1]
    assert snapshot == "2011-12-10"
    assert c1["recency"] == 5            # last purchase 5 Dec -> snapshot 10 Dec
    assert c1["frequency"] == 2          # two distinct invoices
    assert c1["monetary"] == 21.0        # 10 + 10 + 1


def test_rfm_table_is_valid():
    rfm = load_rfm()
    assert rfm["CustomerID"].is_unique
    assert rfm[RFM_FEATURES].notna().all().all()
    assert (rfm["recency"] >= 1).all() and (rfm["frequency"] >= 1).all() and (rfm["monetary"] > 0).all()


def test_metrics_match_saved_model(model, metrics):
    rfm = load_rfm()
    seg = assign_segments(model, rfm)
    assert metrics["chosen_k"] in K_CANDIDATES
    assert [s["customers"] for s in metrics["segments"]] == [int((seg == r).sum()) for r in range(metrics["chosen_k"])]
    Z = model[:-1].transform(rfm[RFM_FEATURES])
    assert math.isclose(silhouette_score(Z, model.named_steps["kmeans"].labels_), metrics["chosen"]["silhouette"])
    assert math.isclose(sum(s["revenue_share"] for s in metrics["segments"]), 1.0)


def test_segments_are_ranked_by_value(metrics):
    medians = [s["median"]["monetary"] for s in metrics["segments"]]
    assert medians[0] == max(medians)                 # rank 0 = most valuable


def test_choose_k_rule():
    ev = [{"k": 2, "silhouette": 0.43}, {"k": 3, "silhouette": 0.338}, {"k": 4, "silhouette": 0.336},
          {"k": 5, "silhouette": 0.318}, {"k": 6, "silhouette": 0.313}]
    assert choose_k(ev) == 4                          # k=2 excluded; 4 is within 0.01 of the best (3)
    ev[2]["silhouette"] = 0.30
    assert choose_k(ev) == 3


@pytest.mark.parametrize("z, name", [
    ([-1.3, 1.7, 1.4], "Champions"), ([-0.9, 0.7, 0.7], "Loyal customers"),
    ([-0.8, -0.4, -0.4], "New or occasional"), ([0.1, 0.4, 0.5], "Regular mid-value"),
    ([0.6, 0.2, 0.3], "At-risk valuable"), ([0.9, -0.8, -0.8], "Lapsed low-value"),
])
def test_naming_rules(z, name):
    assert base_name(np.array(z)) == name


def test_names_unique(model):
    names = segment_names(model)
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("values, expected", [
    (dict(recency=5, frequency=25, monetary=8000.0), "Champions"),
    (dict(recency=250, frequency=1, monetary=150.0), "Lapsed low-value"),
])
def test_scoring_obvious_customers(model, values, expected):
    r = score_customer(model, values)
    assert r["name"] == expected
    assert min(r["distances"], key=r["distances"].get) == expected


@pytest.mark.parametrize("bad", [
    dict(recency=0, frequency=1, monetary=10.0), dict(recency=10, frequency=0, monetary=10.0),
    dict(recency=10, frequency=1, monetary=-5.0), dict(recency=10, frequency=1.5, monetary=10.0),
    dict(recency=10, frequency=1, monetary=float("nan")), dict(recency="x", frequency=1, monetary=10.0),
])
def test_invalid_customers_rejected(model, bad):
    assert validate_customer(bad)
    with pytest.raises(ValueError):
        score_customer(model, bad)


def test_range_warning(metrics):
    assert range_warnings(dict(recency=10, frequency=2, monetary=500.0), metrics["ranges"]) == []
    assert len(range_warnings(dict(recency=10, frequency=500, monetary=500.0), metrics["ranges"])) == 1
