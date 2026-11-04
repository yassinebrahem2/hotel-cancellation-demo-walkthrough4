"""Predictions data freshness check."""

import datetime
from typing import Any

import pandas as pd

from hotel_demo.paths import data_dir


CHECK_NAME = "predictions_freshness"
THRESHOLD_MIN_ROWS = 1
GRACE_DAYS = 2


def run(date: str) -> dict[str, Any]:
    """Run the predictions freshness check for the given date.

    Checks that a predictions parquet file exists for the given business_date
    and contains at least THRESHOLD_MIN_ROWS rows. Allows GRACE_DAYS of lag.

    Appends the result record to data_dir() / "agent-results" / f"{date}.jsonl".
    """
    target_date = datetime.date.fromisoformat(date)
    p_dir = data_dir() / "predictions"
    pred_file = p_dir / f"{date}.parquet"

    if not pred_file.exists():
        # Check grace period
        found = False
        for d in range(1, GRACE_DAYS + 1):
            check_date = target_date - datetime.timedelta(days=d)
            check_file = p_dir / f"{check_date.isoformat()}.parquet"
            if check_file.exists():
                pred_file = check_file
                found = True
                break

        if not found:
            fired = True
            value = 0
            detail = {"error": f"No predictions file found for {date} or within {GRACE_DAYS} days grace"}
        else:
            df = pd.read_parquet(pred_file)
            row_count = len(df)
            fired = row_count < THRESHOLD_MIN_ROWS
            value = row_count
            detail = {"file": pred_file.name, "rows": row_count, "grace_days_used": d}
    else:
        df = pd.read_parquet(pred_file)
        row_count = len(df)
        fired = row_count < THRESHOLD_MIN_ROWS
        value = row_count
        detail = {"file": pred_file.name, "rows": row_count, "grace_days_used": 0}

    record = {
        "check_id": CHECK_NAME,
        "origin": "agent",
        "ran_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "fired": fired,
        "value": float(value),
        "threshold": float(THRESHOLD_MIN_ROWS),
        "category": "data_quality",
        "subject": "hotel-data.predictions",
        "detail": detail,
        "source_ref": "src/hotel_demo/checks/predictions_freshness.py",
        "not_enough_data": False,
    }

    # Write to agent-results
    res_dir = data_dir() / "agent-results"
    res_dir.mkdir(parents=True, exist_ok=True)
    res_file = res_dir / f"{date}.jsonl"
    with res_file.open("a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(record) + "\n")

    return record