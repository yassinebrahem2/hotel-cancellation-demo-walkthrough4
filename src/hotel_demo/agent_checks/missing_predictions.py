"""Missing predictions check - flags business dates with no predictions written."""

import datetime
import json
from typing import Any

from hotel_demo.paths import data_dir

CHECK_NAME = "missing_predictions"
MIN_SAMPLE = 1  # minimum predictions expected for a business date
DEPLOY_STAGE_OUTPUT = "mlflow_model"


def _write_agent_results_line(date: str, record: dict[str, Any]) -> dict[str, Any]:
    """Stamp ran_at (current UTC ISO-8601) and append record to agent-results/<date>.jsonl."""
    ran_at = datetime.datetime.now(datetime.UTC).isoformat()
    stamped: dict[str, Any] = {}
    stamped["check_id"] = CHECK_NAME
    stamped["origin"] = "agent"
    stamped["ran_at"] = ran_at
    for k, v in record.items():
        stamped[k] = v

    ar_dir = data_dir() / "agent-results"
    ar_dir.mkdir(parents=True, exist_ok=True)
    ar_file = ar_dir / f"{date}.jsonl"
    with ar_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(stamped) + "\n")

    return stamped


def run(date: str) -> dict[str, Any]:
    """Run the missing predictions check for the given date.

    Checks if predictions exist for the given business date.
    Fires if no predictions are found (sample < MIN_SAMPLE).
    Never raises — returns a fired result on any error.
    """
    ran_at = datetime.datetime.now(datetime.UTC).isoformat()

    pred_dir = data_dir() / "predictions"
    pred_file = pred_dir / f"{date}.parquet"

    if not pred_file.exists():
        return _write_agent_results_line(
            date,
            {
                "check_id": CHECK_NAME,
                "origin": "agent",
                "ran_at": ran_at,
                "fired": True,
                "value": 0,
                "threshold": MIN_SAMPLE,
                "category": "operational",
                "subject": DEPLOY_STAGE_OUTPUT,
                "model": "hotel-cancellation",
                "detail": {"reason": "no predictions file", "total_predictions": 0, "business_date": date},
                "source_ref": "checks/missing_predictions.py",
                "version": None,
                "not_enough_data": False,
            },
        )

    try:
        import pandas as pd

        df = pd.read_parquet(pred_file)
        total = len(df)
    except Exception as e:
        return _write_agent_results_line(
            date,
            {
                "check_id": CHECK_NAME,
                "origin": "agent",
                "ran_at": ran_at,
                "fired": True,
                "value": None,
                "threshold": MIN_SAMPLE,
                "category": "operational",
                "subject": DEPLOY_STAGE_OUTPUT,
                "model": "hotel-cancellation",
                "detail": {"reason": f"error reading predictions: {e}", "total_predictions": None, "business_date": date},
                "source_ref": "checks/missing_predictions.py",
                "version": None,
                "not_enough_data": False,
            },
        )

    if total < MIN_SAMPLE:
        return _write_agent_results_line(
            date,
            {
                "check_id": CHECK_NAME,
                "origin": "agent",
                "ran_at": ran_at,
                "fired": True,
                "value": total,
                "threshold": MIN_SAMPLE,
                "category": "operational",
                "subject": DEPLOY_STAGE_OUTPUT,
                "model": "hotel-cancellation",
                "detail": {"reason": "below minimum sample", "total_predictions": total, "business_date": date},
                "source_ref": "checks/missing_predictions.py",
                "version": None,
                "not_enough_data": True,
            },
        )

    return _write_agent_results_line(
        date,
        {
            "check_id": CHECK_NAME,
            "origin": "agent",
            "ran_at": ran_at,
            "fired": False,
            "value": total,
            "threshold": MIN_SAMPLE,
            "category": "operational",
            "subject": DEPLOY_STAGE_OUTPUT,
            "model": "hotel-cancellation",
            "detail": {"total_predictions": total, "business_date": date},
            "source_ref": "checks/missing_predictions.py",
            "version": None,
            "not_enough_data": False,
        },
    )


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else str(datetime.date.today()))
