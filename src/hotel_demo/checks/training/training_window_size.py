"""Training window size check - ensures enough training samples."""

import datetime
import json
from typing import Any

import pandas as pd

from hotel_demo.paths import data_dir

CHECK_NAME = "training_window_size"
THRESHOLD = 1000  # minimum training samples
MIN_SAMPLE = 100  # below this, record not_enough_data instead of firing


def _write_agent_result(date: str, record: dict[str, Any]) -> dict[str, Any]:
    """Stamp ran_at (current UTC ISO-8601) and append record to agent-results/<date>.jsonl."""
    ran_at = datetime.datetime.now(datetime.UTC).isoformat()
    stamped: dict[str, Any] = {}
    if "check" in record:
        stamped["check"] = record["check"]
    stamped["ran_at"] = ran_at
    for k, v in record.items():
        if k != "check" and k != "ran_at":
            stamped[k] = v

    res_dir = data_dir() / "agent-results"
    res_dir.mkdir(parents=True, exist_ok=True)
    res_file = res_dir / f"{date}.jsonl"
    with res_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(stamped) + "\n")

    return stamped


def run(date: str) -> dict[str, Any]:
    """Run the training window size check for the given date.

    Checks that the training window (all bookings with arrival_date < date)
    has at least THRESHOLD samples after joining with labels.

    Args:
        date: Business date in ISO format (YYYY-MM-DD).

    Returns:
        Dict with check result.
    """
    check_date = datetime.date.fromisoformat(date)

    # Read bookings and labels
    bookings_path = data_dir() / "bookings"
    labels_path = data_dir() / "labels"

    bookings_files = list(bookings_path.glob("*.parquet"))
    labels_files = list(labels_path.glob("*.parquet"))

    if not bookings_files or not labels_files:
        return {
            "check_id": CHECK_NAME,
            "fired": False,
            "value": 0,
            "threshold": THRESHOLD,
            "category": "data_quality",
            "subject": "training_window",
            "detail": {"error": "no data files found"},
            "source_ref": "src/hotel_demo/checks/training_window_size.py",
            "not_enough_data": True,
        }

    bookings = pd.concat([pd.read_parquet(f) for f in bookings_files], ignore_index=True)
    labels = pd.concat([pd.read_parquet(f) for f in labels_files], ignore_index=True)

    # Filter bookings: arrival_date < check_date (same as train)
    bookings["arrival_date"] = pd.to_datetime(bookings["arrival_date"]).dt.date
    bookings = bookings[bookings["arrival_date"] < check_date]

    # Filter labels: label_date <= check_date (same as train)
    labels["label_date"] = pd.to_datetime(labels["label_date"]).dt.date
    labels = labels[labels["label_date"] <= check_date]

    # Join on booking_id (same as train)
    merged = bookings.merge(labels[["booking_id", "is_canceled", "booking_changes"]], on="booking_id", how="inner")

    n_samples = len(merged)

    not_enough = n_samples < MIN_SAMPLE
    fired = not not_enough and n_samples < THRESHOLD

    return {
        "check_id": CHECK_NAME,
        "fired": fired,
        "value": float(n_samples),
        "threshold": float(THRESHOLD),
        "category": "data_quality",
        "subject": "training_window",
        "detail": {
            "n_bookings_before_cutoff": int(len(bookings)),
            "n_labels_before_cutoff": int(len(labels)),
            "n_merged": int(n_samples),
        },
        "source_ref": "src/hotel_demo/checks/training_window_size.py",
        "not_enough_data": not_enough,
    }


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <date>")
        sys.exit(1)

    date = sys.argv[1]
    result = run(date)
    _write_agent_result(date, result)
    print(f"{CHECK_NAME}: {'FIRED' if result['fired'] else 'OK'} (value={result['value']}, threshold={result['threshold']})")


if __name__ == "__main__":
    main()