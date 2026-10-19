"""Central configuration: paths, dataset source and fixed settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "data" / "raw" / "online_retail.zip"
RAW_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
RAW_SHA256 = "f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b"
RFM_PATH = ROOT / "data" / "processed" / "rfm.csv"
CLEANING_PATH = ROOT / "data" / "processed" / "cleaning_stats.json"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "kmeans_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Stock codes that are services or accounting entries, not products
# (postage, carriage, bank/Amazon fees, discounts, manual adjustments, samples, charity).
NON_PRODUCT_CODES = {"POST", "DOT", "C2", "M", "D", "S", "B", "BANK CHARGES", "AMAZONFEE", "CRUK", "PADS"}

RFM_FEATURES = ["recency", "frequency", "monetary"]
RFM_LABELS = {
    "recency": "Recency (days since last purchase)",
    "frequency": "Frequency (number of orders)",
    "monetary": "Monetary (total spend, £)",
}

K_RANGE = range(2, 11)
# Segments are meant for marketing actions, so a 2-way split (roughly "good" vs
# "bad" customers) is too coarse, and more than 6 segments is hard to act on.
K_CANDIDATES = range(3, 7)
RANDOM_STATE = 42
N_INIT = 20
STABILITY_SEEDS = 10

# Input limits for scoring a new customer in the app.
LIMITS = {"recency": (1, 3650), "frequency": (1, 10_000), "monetary": (0.01, 10_000_000.0)}
