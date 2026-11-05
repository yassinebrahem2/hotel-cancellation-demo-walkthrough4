"""Predictions data schema check."""

import datetime
from typing import Any

import pandas as pd

from hotel_demo.paths import data_dir

CHECK_NAME = "predictions_schema"
GRACE_DAYS = 2

REQUIRED_COLUMNS = {"booking_id", "business_date", "model_version", "p_cancel"}

EXPECTED_DTYPES = {
    "booking_id": "int64",
    "business_date": "object",
    "model_version": "object",
    "p_cancel": "float64",
}


def run(date: str) -> dict[str, Any]:
    """Run the predictions schema check for the given date.

    Validates that the predictions file for the given date has:
    - All required columns present
    - No unexpected columns
    - Correct dtypes

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
            value = 1.0  # schema violation
            detail = {
                "error": f"No predictions file found for {date} or within {GRACE_DAYS} days grace"
            }
        else:
            df = pd.read_parquet(pred_file)
            fired, value, detail = _validate_schema(df, pred_file.name, d)
    else:
        df = pd.read_parquet(pred_file)
        fired, value, detail = _validate_schema(df, pred_file.name, 0)

    record = {
        "check_id": CHECK_NAME,
        "origin": "agent",
        "ran_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "fired": fired,
        "value": float(value),
        "threshold": 0.0,  # 0 = no violations
        "category": "data_quality",
        "subject": "hotel-data.predictions",
        "detail": detail,
        "source_ref": "src/hotel_demo/checks/predictions_schema.py",
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
    """Validate schema of predictions DataFrame."""
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


def _dtypes_compatible(actual: str, expected: str) -> bool:
    """Check if actual dtype is compatible with expected."""
    actual = actual.lower()
    expected = expected.lower()

    if expected in ("int64", "int32", "int16", "int8", "int"):
        return actual.startswith("int")
    if expected in ("float64", "float32", "float"):
        return actual.startswith("float") or actual.startswith("int")
    if expected == "object":
        return actual in ("object", "string", "str")

    return actual == expected
