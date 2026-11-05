"""Segment accuracy alert check.

Evaluates accuracy per hotel and market_segment on matured labels.
"""

import datetime
from typing import Any

import pandas as pd

from hotel_demo.monitoring import write_agent_result_line
from hotel_demo.paths import data_dir

CHECK_NAME = "segment_accuracy"
CHECK_ID = "segment_accuracy"
CATEGORY = "model_performance"
SUBJECT = "hotel-cancellation v1"
MODEL = "hotel-cancellation"
SOURCE_REF = "src/hotel_demo/agent_checks/segment_accuracy.py"
THRESHOLD = 0.60
MIN_SAMPLES_PER_SEGMENT = 20  # minimum labels per segment to evaluate


def _base_record(ran_at: str) -> dict[str, Any]:
    """Build a record with the fields required by RegisterCheck record shape."""
    return {
        "check_id": CHECK_ID,
        "origin": "agent",
        "category": CATEGORY,
        "subject": SUBJECT,
        "model": MODEL,
        "source_ref": SOURCE_REF,
        "ran_at": ran_at,
    }


def _matured_booking_ids(curr_date: datetime.date) -> pd.Series:
    """Return the IDs of the bookings whose arrival date is before curr_date."""
    b_dir = data_dir() / "bookings"
    booking_files = sorted(b_dir.glob("*.parquet"))
    if not booking_files:
        return pd.Series(dtype="int64")
    bookings = pd.concat(
        [
            pd.read_parquet(f, columns=["booking_id", "arrival_date", "hotel", "market_segment"])
            for f in booking_files
        ],
        ignore_index=True,
    )
    arrival = pd.to_datetime(bookings["arrival_date"]).dt.date
    matured = bookings.loc[arrival < curr_date, "booking_id"]
    return matured


def _load_booking_segments() -> pd.DataFrame:
    """Load hotel and market_segment for all bookings."""
    b_dir = data_dir() / "bookings"
    booking_files = sorted(b_dir.glob("*.parquet"))
    if not booking_files:
        return pd.DataFrame(columns=["booking_id", "hotel", "market_segment"])
    bookings = pd.concat(
        [
            pd.read_parquet(f, columns=["booking_id", "hotel", "market_segment"])
            for f in booking_files
        ],
        ignore_index=True,
    )
    return bookings.drop_duplicates(subset=["booking_id"], keep="last")


def run(date: str) -> dict[str, Any]:
    """Run the segment accuracy check for the given date.

    Measures accuracy per hotel and market_segment over matured labels released
    in the last 30 days, joined on booking_id to their predictions and booking
    metadata. Fires if any segment with >= MIN_SAMPLES_PER_SEGMENT labels
    has accuracy < 0.60. Also reports overall accuracy for reference.

    A label is matured when its booking's arrival date (from the bookings files)
    is before the given date. Labels of bookings not in the bookings files are
    excluded. Labels whose arrival_date >= check_date are excluded.

    Appends the result record to data_dir() / "agent-results" / f"{date}.jsonl".
    """
    curr_date = datetime.date.fromisoformat(date)
    start_date = curr_date - datetime.timedelta(days=29)

    # Load matured booking IDs
    matured_ids = _matured_booking_ids(curr_date)
    if len(matured_ids) == 0:
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
            "detail": "no matured bookings",
        }
        return write_agent_result_line(date, record)

    # Load labels in window
    l_dir = data_dir() / "labels"
    label_files = sorted(l_dir.glob("*.parquet"))
    window_labels = pd.DataFrame()
    if label_files:
        all_labels = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)
        all_labels["label_date_dt"] = pd.to_datetime(all_labels["label_date"]).dt.date
        window_labels = all_labels[
            (all_labels["label_date_dt"] >= start_date) & (all_labels["label_date_dt"] <= curr_date)
        ].copy()

    if len(window_labels) == 0:
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
            "detail": "no labels in window",
        }
        return write_agent_result_line(date, record)

    # Filter to matured labels
    window_labels = window_labels[window_labels["booking_id"].isin(matured_ids)]

    if len(window_labels) == 0:
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
            "detail": "no matured labels in window",
        }
        return write_agent_result_line(date, record)

    # Load predictions
    p_dir = data_dir() / "predictions"
    pred_files = sorted(p_dir.glob("*.parquet"))
    all_preds = pd.DataFrame()
    if pred_files:
        all_preds = pd.concat([pd.read_parquet(pf) for pf in pred_files], ignore_index=True)

    if len(all_preds) == 0:
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
            "detail": "no predictions",
        }
        return write_agent_result_line(date, record)

    # Merge labels with predictions
    merged = window_labels[["booking_id", "is_canceled"]].merge(
        all_preds[["booking_id", "p_cancel"]],
        on="booking_id",
        how="inner",
    )
    merged = merged.drop_duplicates(subset=["booking_id"], keep="last")

    if len(merged) == 0:
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
            "detail": "no matched labels and predictions",
        }
        return write_agent_result_line(date, record)

    # Load booking segments
    segments = _load_booking_segments()
    if len(segments) == 0:
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
            "detail": "no booking segments",
        }
        return write_agent_result_line(date, record)

    # Merge with segments
    merged = merged.merge(segments, on="booking_id", how="inner")

    if len(merged) == 0:
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
            "detail": "no matched labels, predictions, and segments",
        }
        return write_agent_result_line(date, record)

    # Compute predictions
    y_true = merged["is_canceled"].astype(int)
    y_pred = (merged["p_cancel"] >= 0.5).astype(int)
    merged["correct"] = (y_true == y_pred).astype(int)

    # Overall accuracy
    overall_total = len(merged)
    overall_correct = int(merged["correct"].sum())
    overall_accuracy = float(overall_correct / overall_total) if overall_total > 0 else None

    # Per-segment accuracy
    segments_detail = {}
    fired = False
    worst_segment = None
    worst_accuracy = 1.0

    for group_col in ["hotel", "market_segment"]:
        group_results = (
            merged.groupby(group_col, observed=False)
            .agg(
                total=("correct", "size"),
                correct=("correct", "sum"),
            )
            .reset_index()
        )

        for _, row in group_results.iterrows():
            segment_name = str(row[group_col])
            total = int(row["total"])
            correct = int(row["correct"])
            accuracy = float(correct / total) if total > 0 else None

            key = f"{group_col}:{segment_name}"
            segments_detail[key] = {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "enough_data": total >= MIN_SAMPLES_PER_SEGMENT,
            }

            # Fire if segment has enough data and accuracy below threshold
            if total >= MIN_SAMPLES_PER_SEGMENT and accuracy is not None and accuracy < THRESHOLD:
                fired = True
                if accuracy < worst_accuracy:
                    worst_accuracy = accuracy
                    worst_segment = key

    # Overall value is the worst segment accuracy (or overall if no segment fires)
    value = worst_accuracy if fired else overall_accuracy

    detail = {
        "overall": {
            "total": overall_total,
            "correct": overall_correct,
            "accuracy": overall_accuracy,
        },
        "segments": segments_detail,
        "min_samples_per_segment": MIN_SAMPLES_PER_SEGMENT,
    }
    if worst_segment:
        detail["worst_segment"] = worst_segment

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
