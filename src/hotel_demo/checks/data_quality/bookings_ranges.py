"""Bookings data ranges check."""

import datetime
from typing import Any

import pandas as pd

from hotel_demo.paths import data_dir


CHECK_NAME = "bookings_ranges"
GRACE_DAYS = 2

# Expected value ranges for numeric columns (based on data summary)
RANGES = {
    "lead_time": (0, 737),
    "arrival_month": (1, 12),
    "arrival_week": (1, 53),
    "stays_in_weekend_nights": (0, 19),
    "stays_in_week_nights": (0, 50),
    "adults": (1, 55),  # 0 is invalid for a booking
    "children": (0, 10),
    "babies": (0, 10),
    "is_repeated_guest": (0, 1),
    "previous_cancellations": (0, 26),
    "previous_bookings_not_canceled": (0, 38),
    "booking_changes": (0, 0),  # all zeros in data
    "adr": (0.0, 5400.0),
    "total_of_special_requests": (0, 5),
}


def run(date: str) -> dict[str, Any]:
    """Run the bookings ranges check for the given date.

    Validates that numeric columns fall within expected ranges.

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
            detail = {"error": f"No bookings file found for {date} or within {GRACE_DAYS} days grace"}
        else:
            df = pd.read_parquet(booking_file)
            fired, value, detail = _check_ranges(df, booking_file.name, d)
    else:
        df = pd.read_parquet(booking_file)
        fired, value, detail = _check_ranges(df, booking_file.name, 0)

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
        "source_ref": "src/hotel_demo/checks/bookings_ranges.py",
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


def _check_ranges(df: pd.DataFrame, filename: str, grace_days: int) -> tuple[bool, float, dict[str, Any]]:
    """Check value ranges for numeric columns."""
    violations = []
    range_info = {}
    total_rows = len(df)

    for col, (min_val, max_val) in RANGES.items():
        if col in df.columns:
            # Only check non-null values
            non_null = df[col].dropna()
            if len(non_null) > 0:
                actual_min = non_null.min()
                actual_max = non_null.max()
                out_of_range = non_null[(non_null < min_val) | (non_null > max_val)]
                out_count = len(out_of_range)

                range_info[col] = {
                    "expected_min": min_val,
                    "expected_max": max_val,
                    "actual_min": float(actual_min) if pd.api.types.is_numeric_dtype(non_null) else str(actual_min),
                    "actual_max": float(actual_max) if pd.api.types.is_numeric_dtype(non_null) else str(actual_max),
                    "out_of_range_count": int(out_count),
                    "non_null_count": int(len(non_null)),
                }

                if out_count > 0:
                    violations.append(
                        f"Column {col}: {out_count} values outside range [{min_val}, {max_val}] "
                        f"(min={actual_min}, max={actual_max})"
                    )
            else:
                range_info[col] = {
                    "expected_min": min_val,
                    "expected_max": max_val,
                    "actual_min": None,
                    "actual_max": None,
                    "out_of_range_count": 0,
                    "non_null_count": 0,
                }
        else:
            range_info[col] = {
                "expected_min": min_val,
                "expected_max": max_val,
                "actual_min": None,
                "actual_max": None,
                "out_of_range_count": total_rows,
                "non_null_count": 0,
            }
            violations.append(f"Column {col}: missing from data")

    fired = len(violations) > 0
    value = float(len(violations))
    detail = {
        "file": filename,
        "grace_days_used": grace_days,
        "rows": total_rows,
        "range_checks": range_info,
        "violations": violations,
    }

    return fired, value, detail