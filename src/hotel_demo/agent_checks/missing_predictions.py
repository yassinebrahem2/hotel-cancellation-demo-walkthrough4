"""Missing predictions check - flags business dates with no predictions written."""

import datetime
from typing import Any

import pandas as pd

from hotel_demo.paths import data_dir

CHECK_NAME = "missing_predictions"
MIN_SAMPLE = 1  # minimum predictions expected for a business date


def run(date: str) -> dict[str, Any]:
    """Run the missing predictions check for the given date.

    Checks if predictions exist for the given business date.
    Fires if no predictions are found (sample < MIN_SAMPLE).
    """
    ran_at = datetime.datetime.now(datetime.UTC).isoformat()

    pred_dir = data_dir() / "predictions"
    pred_file = pred_dir / f"{date}.parquet"

    if not pred_file.exists():
        return {
            "check_id": CHECK_NAME,
            "origin": "agent",
            "ran_at": ran_at,
            "fired": True,
            "value": 0,
            "threshold": MIN_SAMPLE,
            "category": "operational",
            "subject": f"predictions for {date}",
            "model": "hotel-cancellation",
            "detail": {"reason": "no predictions file", "total_predictions": 0},
            "source_ref": "checks/missing_predictions.py",
            "version": None,
            "not_enough_data": False,
        }

    df = pd.read_parquet(pred_file)
    total = len(df)

    if total < MIN_SAMPLE:
        return {
            "check_id": CHECK_NAME,
            "origin": "agent",
            "ran_at": ran_at,
            "fired": True,
            "value": total,
            "threshold": MIN_SAMPLE,
            "category": "operational",
            "subject": f"predictions for {date}",
            "model": "hotel-cancellation",
            "detail": {"reason": "below minimum sample", "total_predictions": total},
            "source_ref": "checks/missing_predictions.py",
            "version": None,
            "not_enough_data": True,
        }

    return {
        "check_id": CHECK_NAME,
        "origin": "agent",
        "ran_at": ran_at,
        "fired": False,
        "value": total,
        "threshold": MIN_SAMPLE,
        "category": "operational",
        "subject": f"predictions for {date}",
        "model": "hotel-cancellation",
        "detail": {"total_predictions": total},
        "source_ref": "checks/missing_predictions.py",
        "version": None,
        "not_enough_data": False,
    }
