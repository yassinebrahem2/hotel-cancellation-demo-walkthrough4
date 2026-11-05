"""Categorical drift check using Population Stability Index (PSI).

Evaluates drift for categorical features: hotel, market_segment, meal, country,
deposit_type, customer_type, distribution_channel.
"""

import datetime
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from hotel_demo.monitoring import write_agent_result_line
from hotel_demo.paths import data_dir, tracking_uri

CHECK_NAME = "categorical_drift"
CHECK_ID = "categorical_drift"
CATEGORY = "drift"
SUBJECT = "hotel-cancellation v1"
MODEL = "hotel-cancellation"
SOURCE_REF = "src/hotel_demo/agent_checks/categorical_drift.py"
THRESHOLD = 0.2  # PSI > 0.2 indicates significant drift
MIN_SAMPLES = 30  # minimum samples per category per window


CATEGORICAL_FEATURES = [
    "hotel",
    "market_segment",
    "meal",
    "country",
    "deposit_type",
    "customer_type",
    "distribution_channel",
]


def _psi(expected: pd.Series, actual: pd.Series) -> float | None:
    """Compute Population Stability Index between two categorical distributions.

    Args:
        expected: Reference distribution (training)
        actual: Current distribution (serving)

    Returns:
        PSI value, or None if computation not possible
    """
    # Get value counts
    expected_counts = expected.value_counts(normalize=True)
    actual_counts = actual.value_counts(normalize=True)

    # Align categories (union of both)
    all_categories = set(expected_counts.index) | set(actual_counts.index)

    if len(all_categories) == 0:
        return None

    psi_sum = 0.0
    for cat in all_categories:
        e = expected_counts.get(cat, 0.0001)  # small epsilon for unseen
        a = actual_counts.get(cat, 0.0001)
        psi_sum += (a - e) * np.log(a / e)

    return float(psi_sum)


def _load_training_distributions() -> dict[str, pd.Series]:
    """Load training distributions for categorical features from MLflow."""
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()

    # Get the production model version
    try:
        prod_versions = client.get_latest_versions("hotel-cancellation", stages=["Production"])
        if not prod_versions:
            return {}
        model_version = prod_versions[0].version
    except Exception:
        return {}

    # Get the run that produced this model version
    client.get_model_version("hotel-cancellation", model_version)

    # Try to load training data distributions from the run artifacts
    # We'll compute from the training data directly as a fallback
    return {}


def _load_serving_window(date: str, window_days: int = 7) -> pd.DataFrame:
    """Load serving (current) data for the last window_days."""
    curr_date = datetime.date.fromisoformat(date)
    start_date = curr_date - datetime.timedelta(days=window_days - 1)

    b_dir = data_dir() / "bookings"
    booking_files = sorted(b_dir.glob("*.parquet"))
    if not booking_files:
        return pd.DataFrame()

    all_bookings = pd.concat(
        [
            pd.read_parquet(f, columns=["booking_date", *CATEGORICAL_FEATURES])
            for f in booking_files
        ],
        ignore_index=True,
    )
    all_bookings["booking_date_dt"] = pd.to_datetime(all_bookings["booking_date"]).dt.date
    window_bookings = all_bookings[
        (all_bookings["booking_date_dt"] >= start_date)
        & (all_bookings["booking_date_dt"] <= curr_date)
    ].copy()

    return window_bookings


def _load_training_data(cutoff_date: str) -> pd.DataFrame:
    """Load training data up to cutoff_date."""
    cutoff = datetime.date.fromisoformat(cutoff_date)
    b_dir = data_dir() / "bookings"
    booking_files = sorted(b_dir.glob("*.parquet"))
    if not booking_files:
        return pd.DataFrame()

    all_bookings = pd.concat(
        [
            pd.read_parquet(f, columns=["booking_date", *CATEGORICAL_FEATURES])
            for f in booking_files
        ],
        ignore_index=True,
    )
    all_bookings["booking_date_dt"] = pd.to_datetime(all_bookings["booking_date"]).dt.date
    training_bookings = all_bookings[all_bookings["booking_date_dt"] < cutoff].copy()

    return training_bookings


def run(date: str) -> dict[str, Any]:
    """Run the categorical drift check for the given date.

    Computes PSI for each categorical feature between training distribution
    (bookings before the model's training cutoff) and serving distribution
    (bookings in the last 7 days up to date). Fires if any feature has PSI > 0.2.

    Appends the result record to data_dir() / "agent-results" / f"{date}.jsonl".
    """
    datetime.date.fromisoformat(date)

    # Load training data - use a fixed cutoff that matches the production model
    # For now, use the same logic as drift check: train on data before 2016-07-01
    training_cutoff = datetime.date(2016, 7, 1)
    training_data = _load_training_data(training_cutoff.isoformat())

    # Load serving window (last 7 days)
    serving_data = _load_serving_window(date, window_days=7)

    if len(training_data) == 0:
        record = {
            "check_id": CHECK_ID,
            "origin": "agent",
            "category": CATEGORY,
            "subject": SUBJECT,
            "model": MODEL,
            "source_ref": SOURCE_REF,
            "fired": False,
            "value": None,
            "threshold": float(THRESHOLD),
            "detail": "no training data",
        }
        return write_agent_result_line(date, record)

    if len(serving_data) == 0:
        record = {
            "check_id": CHECK_ID,
            "origin": "agent",
            "category": CATEGORY,
            "subject": SUBJECT,
            "model": MODEL,
            "source_ref": SOURCE_REF,
            "fired": False,
            "value": None,
            "threshold": float(THRESHOLD),
            "detail": "no serving data",
        }
        return write_agent_result_line(date, record)

    # Compute PSI for each categorical feature
    features_detail = {}
    fired = False
    max_psi = 0.0
    worst_feature = None

    for feat in CATEGORICAL_FEATURES:
        if feat not in training_data.columns or feat not in serving_data.columns:
            features_detail[feat] = None
            continue

        train_series = training_data[feat].dropna()
        serve_series = serving_data[feat].dropna()

        if len(train_series) < MIN_SAMPLES or len(serve_series) < MIN_SAMPLES:
            features_detail[feat] = {
                "psi": None,
                "train_samples": len(train_series),
                "serve_samples": len(serve_series),
                "enough_data": False,
            }
            continue

        psi_val = _psi(train_series, serve_series)
        enough_data = True

        features_detail[feat] = {
            "psi": psi_val,
            "train_samples": len(train_series),
            "serve_samples": len(serve_series),
            "enough_data": enough_data,
        }

        if psi_val is not None and psi_val > max_psi:
            max_psi = psi_val
            worst_feature = feat

        if psi_val is not None and psi_val > THRESHOLD:
            fired = True

    value = max_psi if max_psi > 0 else None

    detail = {
        "features": features_detail,
        "days_evaluated": [date],
        "training_cutoff": training_cutoff.isoformat(),
        "serving_window_days": 7,
    }
    if worst_feature:
        detail["worst_feature"] = worst_feature

    record = {
        "check_id": CHECK_ID,
        "origin": "agent",
        "category": CATEGORY,
        "subject": SUBJECT,
        "model": MODEL,
        "source_ref": SOURCE_REF,
        "fired": fired,
        "value": value,
        "threshold": float(THRESHOLD),
        "detail": detail,
    }

    return write_agent_result_line(date, record)
