"""Bookings data schema check."""

import datetime
from typing import Any

import pandas as pd

from hotel_demo.features import BOOKING_COLUMNS
from hotel_demo.paths import data_dir

CHECK_NAME = "bookings_schema"
GRACE_DAYS = 2


# Expected dtypes for critical columns
# Note: agent and company are stored as float64 in raw parquet (nullable ints with many nulls)
# but get converted to strings in prepare_features. Accept both.
EXPECTED_DTYPES = {
    "booking_id": "int64",
    "booking_date": "object",  # date stored as string/object
    "arrival_date": "object",
    "hotel": "object",
    "lead_time": "int64",
    "arrival_month": "int64",
    "arrival_week": "int64",
    "stays_in_weekend_nights": "int64",
    "stays_in_week_nights": "int64",
    "adults": "int64",
    "children": "float64",
    "babies": "int64",
    "meal": "object",
    "country": "object",
    "market_segment": "object",
    "distribution_channel": "object",
    "is_repeated_guest": "int64",
    "previous_cancellations": "int64",
    "previous_bookings_not_canceled": "int64",
    "reserved_room_type": "object",
    "booking_changes": "int64",
    "deposit_type": "object",
    "agent": ["float64", "object"],  # float64 in raw, string after prepare_features
    "company": ["float64", "object"],  # float64 in raw, string after prepare_features
    "customer_type": "object",
    "adr": "float64",
    "total_of_special_requests": "int64",
}

# Columns that must be present (subset of BOOKING_COLUMNS)
REQUIRED_COLUMNS = set(BOOKING_COLUMNS)


def run(date: str) -> dict[str, Any]:
    """Run the bookings schema check for the given date.

    Validates that the bookings file for the given date has:
    - All required columns present
    - No unexpected columns
    - Correct dtypes for critical columns

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
            value = 1.0  # schema violation
            detail = {
                "error": f"No bookings file found for {date} or within {GRACE_DAYS} days grace"
            }
        else:
            df = pd.read_parquet(booking_file)
            fired, value, detail = _validate_schema(df, booking_file.name, d)
    else:
        df = pd.read_parquet(booking_file)
        fired, value, detail = _validate_schema(df, booking_file.name, 0)

    record = {
        "check_id": CHECK_NAME,
        "origin": "agent",
        "ran_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "fired": fired,
        "value": float(value),
        "threshold": 0.0,  # 0 = no violations
        "category": "data_quality",
        "subject": "hotel-data.bookings",
        "detail": detail,
        "source_ref": "src/hotel_demo/checks/bookings_schema.py",
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


def _validate_schema(
    df: pd.DataFrame, filename: str, grace_days: int
) -> tuple[bool, float, dict[str, Any]]:
    """Validate schema of bookings DataFrame."""
    violations = []
    actual_columns = set(df.columns)

    # Check for missing required columns
    missing = REQUIRED_COLUMNS - actual_columns
    if missing:
        violations.append(f"Missing columns: {sorted(missing)}")

    # Check for unexpected columns
    unexpected = actual_columns - REQUIRED_COLUMNS
    if unexpected:
        violations.append(f"Unexpected columns: {sorted(unexpected)}")

    # Check dtypes for columns that exist
    for col, expected_dtype in EXPECTED_DTYPES.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            # Normalize dtype comparison
            if not _dtypes_compatible(actual_dtype, expected_dtype):
                violations.append(f"Column {col}: expected {expected_dtype}, got {actual_dtype}")

    fired = len(violations) > 0
    value = float(len(violations))
    detail = {
        "file": filename,
        "grace_days_used": grace_days,
        "rows": len(df),
        "columns": sorted(df.columns.tolist()),
        "violations": violations,
    }

    return fired, value, detail


def _dtypes_compatible(actual: str, expected) -> bool:
    """Check if actual dtype is compatible with expected."""
    # Handle list of expected dtypes
    if isinstance(expected, list):
        return any(_dtypes_compatible(actual, e) for e in expected)

    # Normalize
    actual = actual.lower()
    expected = expected.lower()

    # Int compatibility
    if expected in ("int64", "int32", "int16", "int8", "int"):
        return actual.startswith("int")
    if expected in ("float64", "float32", "float"):
        return actual.startswith("float") or actual.startswith("int")
    if expected == "object":
        return actual in ("object", "string", "str")

    return actual == expected
