"""Training label maturity check - ensures labels are matured per model settings."""

import datetime
import json
from typing import Any

import pandas as pd

from hotel_demo.paths import data_dir

# Label maturity in days from the handoff
LABEL_MATURITY_DAYS = 0

CHECK_NAME = "training_label_maturity"
THRESHOLD = float(LABEL_MATURITY_DAYS)  # labels must be at least this many days old
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
    """Run the training label maturity check for the given date.

    Checks that labels used in training (label_date <= cutoff) have
    label_date <= cutoff - label_maturity_days. With label_maturity=0,
    this means all labels with label_date <= cutoff are valid.

    Args:
        date: Business date in ISO format (YYYY-MM-DD).

    Returns:
        Dict with check result.
    """
    check_date = datetime.date.fromisoformat(date)
    maturity_cutoff = check_date - datetime.timedelta(days=LABEL_MATURITY_DAYS)

    # Read bookings and labels
    bookings_path = data_dir() / "bookings"
    labels_path = data_dir() / "labels"

    bookings_files = list(bookings_path.glob("*.parquet"))
    labels_files = list(labels_path.glob("*.parquet"))

    if not bookings_files or not labels_files:
        return {
            "check_id": CHECK_NAME,
            "fired": False,
            "value": 0.0,
            "threshold": THRESHOLD,
            "category": "data_quality",
            "subject": "training_window",
            "detail": {"error": "no data files found"},
            "source_ref": "src/hotel_demo/checks/training_label_maturity.py",
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

    # Check label maturity: label_date <= check_date - label_maturity_days
    immature_labels = labels[labels["label_date"] > maturity_cutoff]
    n_immature = len(immature_labels)
    n_total_labels = len(labels)

    # Join on booking_id (same as train)
    merged = bookings.merge(
        labels[["booking_id", "is_canceled", "booking_changes"]], on="booking_id", how="inner"
    )
    n_merged = len(merged)

    if n_merged < MIN_SAMPLE:
        return {
            "check_id": CHECK_NAME,
            "fired": False,
            "value": float(n_immature),
            "threshold": THRESHOLD,
            "category": "data_quality",
            "subject": "training_window",
            "detail": {
                "n_merged": int(n_merged),
                "n_total_labels": int(n_total_labels),
                "n_immature_labels": int(n_immature),
                "label_maturity_days": LABEL_MATURITY_DAYS,
                "error": "not enough merged samples",
            },
            "source_ref": "src/hotel_demo/checks/training_label_maturity.py",
            "not_enough_data": True,
        }

    # Fire if any immature labels are in the training set
    fired = n_immature > 0

    return {
        "check_id": CHECK_NAME,
        "fired": fired,
        "value": float(n_immature),
        "threshold": THRESHOLD,
        "category": "data_quality",
        "subject": "training_window",
        "detail": {
            "n_merged": int(n_merged),
            "n_total_labels": int(n_total_labels),
            "n_immature_labels": int(n_immature),
            "label_maturity_days": LABEL_MATURITY_DAYS,
            "maturity_cutoff": maturity_cutoff.isoformat(),
        },
        "source_ref": "src/hotel_demo/checks/training_label_maturity.py",
        "not_enough_data": False,
    }


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <date>")
        sys.exit(1)

    date = sys.argv[1]
    result = run(date)
    _write_agent_result(date, result)
    print(
        f"{CHECK_NAME}: {'FIRED' if result['fired'] else 'OK'} (immature_labels={result['value']}, threshold={result['threshold']})"
    )


if __name__ == "__main__":
    main()
