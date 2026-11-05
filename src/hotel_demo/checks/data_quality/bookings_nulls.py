"""Bookings data nulls check."""

import datetime
from typing import Any

import pandas as pd

from hotel_demo.paths import data_dir

CHECK_NAME = "bookings_nulls"
GRACE_DAYS = 2

# Critical columns that should have low null rates
CRITICAL_COLUMNS = {
    "booking_id": 0.0,  # zero tolerance
    "booking_date": 0.0,  # zero tolerance
    "arrival_date": 0.0,  # zero tolerance
    "hotel": 0.0,  # zero tolerance
    "lead_time": 0.0,  # zero tolerance
    "adr": 0.0,  # zero tolerance
    "country": 0.05,  # 5% max null (312 nulls in data)
    "meal": 0.05,  # 5% max null (624 nulls in data)
    "market_segment": 0.01,  # 1% max null (2 nulls in data)
    "distribution_channel": 0.01,  # 1% max null (5 nulls in data)
    "children": 0.01,  # 1% max null (4 nulls in data)
}

# Columns with known high nulls - track but don't fire
HIGH_NULL_COLUMNS = {"agent", "company"}


def run(date: str) -> dict[str, Any]:
    """Run the bookings nulls check for the given date.

    Validates that critical columns have null rates below thresholds.
    Tracks high-null columns (agent, company) for visibility but doesn't fire.

    Appends the result record to data_dir() / "agent-results" / f"{date}.jsonl".
    """
    target_date = datetime.date.fromisoformat(date)
    b_dir = data_dir() / "bookings"
    booking_file = b_dir / f"{date}.parquet"

    if not booking_file.exists():
        # Check grace period
        found = False
        for d in range(1, GRACE_DAYS + 1):
            check_date = target_date - datetime.timedelta(days=d)
            check_file = b_dir / f"{check_date.isoformat()}.parquet"
            if check_file.exists():
                booking_file = check_file
                found = True
                break

        if not found:
            fired = True
            value = 1.0
            detail = {
                "error": f"No bookings file found for {date} or within {GRACE_DAYS} days grace"
            }
        else:
            df = pd.read_parquet(booking_file)
            fired, value, detail = _check_nulls(df, booking_file.name, d)
    else:
        df = pd.read_parquet(booking_file)
        fired, value, detail = _check_nulls(df, booking_file.name, 0)

    record = {
        "check_id": CHECK_NAME,
        "origin": "agent",
        "ran_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "fired": fired,
        "value": float(value),
        "threshold": 0.0,  # max allowed violations = 0
        "category": "data_quality",
        "subject": "hotel-data.bookings",
        "detail": detail,
        "source_ref": "src/hotel_demo/checks/data_quality/bookings_nulls.py",
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


def _check_nulls(
    df: pd.DataFrame, filename: str, grace_days: int
) -> tuple[bool, float, dict[str, Any]]:
    """Check null rates in critical columns."""
    violations = []
    null_rates = {}
    total_rows = len(df)

    for col, max_null_rate in CRITICAL_COLUMNS.items():
        if col in df.columns:
            null_count = df[col].isna().sum()
            null_rate = null_count / total_rows if total_rows > 0 else 0
            null_rates[col] = {
                "null_count": int(null_count),
                "null_rate": float(null_rate),
                "threshold": max_null_rate,
            }
            if null_rate > max_null_rate:
                violations.append(
                    f"Column {col}: null rate {null_rate:.4f} > threshold {max_null_rate} "
                    f"({null_count}/{total_rows} nulls)"
                )
        else:
            violations.append(f"Column {col}: missing from data")
            null_rates[col] = {
                "null_count": total_rows,
                "null_rate": 1.0,
                "threshold": max_null_rate,
            }

    # Track high-null columns for visibility
    high_null_info = {}
    for col in HIGH_NULL_COLUMNS:
        if col in df.columns:
            null_count = df[col].isna().sum()
            null_rate = null_count / total_rows if total_rows > 0 else 0
            high_null_info[col] = {"null_count": int(null_count), "null_rate": float(null_rate)}
        else:
            high_null_info[col] = {"null_count": total_rows, "null_rate": 1.0}

    fired = len(violations) > 0
    value = float(len(violations))
    detail = {
        "file": filename,
        "grace_days_used": grace_days,
        "rows": total_rows,
        "critical_nulls": null_rates,
        "high_null_columns": high_null_info,
        "violations": violations,
    }

    return fired, value, detail
