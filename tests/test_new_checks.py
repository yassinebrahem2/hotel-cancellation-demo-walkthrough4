"""Tests for new agent checks: segment_accuracy and categorical_drift."""

import datetime
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from hotel_demo.paths import data_dir
from hotel_demo.pipeline import deploy, train


@pytest.fixture
def checks_env(tmp_path, monkeypatch):
    data_target = tmp_path / "data"
    mlflow_target = tmp_path / "mlflow"
    fixture_data = Path(__file__).parent / "fixtures" / "data"

    shutil.copytree(fixture_data, data_target)
    mlflow_target.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOTEL_DEMO_DATA", str(data_target))
    monkeypatch.setenv("HOTEL_DEMO_MLFLOW", str(mlflow_target))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    v1 = train(cutoff="2026-01-10", min_child_samples=5)
    deploy(version=v1)

    return {"data": data_target, "mlflow": mlflow_target, "version": v1}


def _read_agent_result_lines(date: str) -> list[dict]:
    res_file = data_dir() / "agent-results" / f"{date}.jsonl"
    assert res_file.exists(), f"Agent results file {res_file} does not exist"
    lines = []
    with res_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    return lines


def _assert_utc_iso_timestamp(
    ts: str,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
) -> None:
    dt = datetime.datetime.fromisoformat(ts)
    assert dt.tzinfo is not None, f"Expected timezone-aware timestamp, got {ts}"
    assert dt.utcoffset() == datetime.timedelta(0), f"Expected UTC offset 0, got {dt.utcoffset()}"
    if start is not None:
        assert start <= dt, f"Timestamp {dt} earlier than start {start}"
    if end is not None:
        assert dt <= end, f"Timestamp {dt} later than end {end}"


def _write_segment_accuracy_case(
    check_date: str,
    label_date: str,
    booking_rows: list[dict],
    pred_rows: list[dict],
) -> None:
    """Helper to write bookings, labels, predictions for segment accuracy test.

    booking_rows: list of dict with booking_id, arrival_date, hotel, market_segment
    pred_rows: list of dict with booking_id, p_cancel
    """
    d = data_dir()

    # Write bookings
    b_df = pd.DataFrame(booking_rows)
    b_df["booking_date"] = pd.to_datetime(b_df["arrival_date"]) - pd.to_timedelta(b_df.get("lead_time", 30), unit="D")
    b_df["booking_date"] = b_df["booking_date"].dt.strftime("%Y-%m-%d")
    # Add required columns with defaults
    for col in ["lead_time", "arrival_month", "arrival_week", "stays_in_weekend_nights",
                "stays_in_week_nights", "adults", "children", "babies", "meal", "country",
                "distribution_channel", "is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled", "reserved_room_type", "booking_changes",
                "deposit_type", "agent", "company", "customer_type", "adr",
                "total_of_special_requests"]:
        if col not in b_df.columns:
            if col in ["adults", "children", "babies", "lead_time", "arrival_month",
                       "arrival_week", "stays_in_weekend_nights", "stays_in_week_nights",
                       "is_repeated_guest", "previous_cancellations",
                       "previous_bookings_not_canceled", "booking_changes",
                       "total_of_special_requests"]:
                b_df[col] = 1 if col in ["adults", "lead_time"] else 0
            else:
                b_df[col] = "BB" if col == "meal" else ("PRT" if col == "country" else
                    ("TA/TO" if col == "distribution_channel" else
                    ("No Deposit" if col == "deposit_type" else
                    ("A" if col == "reserved_room_type" else
                    ("Transient" if col == "customer_type" else None)))))
    b_df.to_parquet(d / "bookings" / "2026-01-11.parquet", index=False)

    # Write labels
    l_df = pd.DataFrame({
        "booking_id": [r["booking_id"] for r in booking_rows],
        "is_canceled": [r["is_canceled"] for r in booking_rows],
        "label_date": label_date,
        "booking_changes": 0,
    })
    l_df.to_parquet(d / "labels" / f"{label_date}.parquet", index=False)

    # Write predictions
    pred_dir = d / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    p_df = pd.DataFrame(pred_rows)
    p_df["business_date"] = "2026-01-11"
    p_df["model_version"] = "1"
    p_df.to_parquet(d / "predictions" / "2026-01-11.parquet", index=False)


def test_segment_accuracy_check_quiet(checks_env):
    from hotel_demo.agent_checks import segment_accuracy

    assert segment_accuracy.CHECK_NAME == "segment_accuracy"
    assert segment_accuracy.THRESHOLD == 0.60

    # Create predictions that match labels perfectly
    pred_dir = data_dir() / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Use labels from fixture
    l_df = pd.read_parquet(data_dir() / "labels" / "2026-01-12.parquet")
    pred_rows = []
    for _, row in l_df.iterrows():
        b_id = int(row["booking_id"])
        is_canc = int(row["is_canceled"])
        p_canc = 0.9 if is_canc == 1 else 0.1
        pred_rows.append({
            "booking_id": b_id,
            "business_date": "2026-01-11",
            "model_version": "1",
            "p_cancel": p_canc,
        })
    pd.DataFrame(pred_rows).to_parquet(pred_dir / "2026-01-11.parquet", index=False)

    # Use fixture data which has sufficient samples per segment
    # Fixtures have labels up to 2026-01-13, so use a date within window
    start = datetime.datetime.now(datetime.UTC)
    res = segment_accuracy.run("2026-01-13")
    end = datetime.datetime.now(datetime.UTC)

    assert set(res.keys()) == {"check", "ran_at", "fired", "value", "threshold", "detail"}
    assert res["check"] == "segment_accuracy"
    assert res["threshold"] == 0.60
    _assert_utc_iso_timestamp(res["ran_at"], start, end)

    lines = _read_agent_result_lines("2026-01-13")
    assert len(lines) == 1
    assert lines[0] == res

    # Check detail structure
    assert "overall" in res["detail"]
    assert "segments" in res["detail"]
    assert "min_samples_per_segment" in res["detail"]


def test_segment_accuracy_check_fires_on_bad_segment(checks_env):
    from hotel_demo.agent_checks import segment_accuracy

    # Create a case where one segment has bad accuracy
    # Hotel: City Hotel - 30 bookings, all predictions wrong
    # Hotel: Resort Hotel - 30 bookings, all predictions correct
    booking_rows = []
    pred_rows = []
    booking_id = 10000

    # City Hotel - 30 bookings, arrival before check date (matured)
    for i in range(30):
        booking_rows.append({
            "booking_id": booking_id,
            "arrival_date": "2026-01-10",  # before 2026-01-15
            "hotel": "City Hotel",
            "market_segment": "Direct",
            "is_canceled": 1 if i % 2 == 0 else 0,
            "lead_time": 30,
        })
        # Wrong prediction
        pred_rows.append({
            "booking_id": booking_id,
            "p_cancel": 0.1 if i % 2 == 0 else 0.9,  # inverted
        })
        booking_id += 1

    # Resort Hotel - 30 bookings, arrival before check date (matured)
    for i in range(30):
        booking_rows.append({
            "booking_id": booking_id,
            "arrival_date": "2026-01-10",
            "hotel": "Resort Hotel",
            "market_segment": "Direct",
            "is_canceled": 1 if i % 2 == 0 else 0,
            "lead_time": 30,
        })
        # Correct prediction
        pred_rows.append({
            "booking_id": booking_id,
            "p_cancel": 0.9 if i % 2 == 0 else 0.1,
        })
        booking_id += 1

    _write_segment_accuracy_case(
        check_date="2026-01-15",
        label_date="2026-01-12",
        booking_rows=booking_rows,
        pred_rows=pred_rows,
    )

    start = datetime.datetime.now(datetime.UTC)
    res = segment_accuracy.run("2026-01-15")
    end = datetime.datetime.now(datetime.UTC)

    assert res["fired"] is True
    assert res["value"] < 0.60
    assert "worst_segment" in res["detail"]
    assert res["detail"]["worst_segment"] == "hotel:City Hotel"
    assert res["detail"]["segments"]["hotel:City Hotel"]["accuracy"] < 0.60
    assert res["detail"]["segments"]["hotel:Resort Hotel"]["accuracy"] == 1.0
    _assert_utc_iso_timestamp(res["ran_at"], start, end)


def test_segment_accuracy_check_not_enough_data_per_segment(checks_env):
    from hotel_demo.agent_checks import segment_accuracy

    # Clear fixture labels to avoid conflicts
    for f in (data_dir() / "labels").glob("*.parquet"):
        f.unlink()

    # Only 10 bookings per segment (below MIN_SAMPLES_PER_SEGMENT=20)
    # Use different market_segment for each hotel so they don't aggregate
    booking_rows = []
    pred_rows = []
    booking_id = 20000

    for i in range(10):
        booking_rows.append({
            "booking_id": booking_id,
            "arrival_date": "2026-02-01",  # before check date
            "hotel": "City Hotel",
            "market_segment": "Online TA",
            "is_canceled": 1,
            "lead_time": 30,
        })
        pred_rows.append({
            "booking_id": booking_id,
            "p_cancel": 0.1,  # wrong prediction
        })
        booking_id += 1

    for i in range(10):
        booking_rows.append({
            "booking_id": booking_id,
            "arrival_date": "2026-02-01",
            "hotel": "Resort Hotel",
            "market_segment": "Offline TA/TO",
            "is_canceled": 1,
            "lead_time": 30,
        })
        pred_rows.append({
            "booking_id": booking_id,
            "p_cancel": 0.1,  # wrong prediction
        })
        booking_id += 1

    _write_segment_accuracy_case(
        check_date="2026-02-15",
        label_date="2026-02-10",
        booking_rows=booking_rows,
        pred_rows=pred_rows,
    )

    res = segment_accuracy.run("2026-02-15")

    # Should not fire because segments don't have enough data
    assert res["fired"] is False
    # But should report the segments with enough_data=False
    assert res["detail"]["segments"]["hotel:City Hotel"]["enough_data"] is False
    assert res["detail"]["segments"]["hotel:Resort Hotel"]["enough_data"] is False
    assert res["detail"]["segments"]["market_segment:Online TA"]["enough_data"] is False
    assert res["detail"]["segments"]["market_segment:Offline TA/TO"]["enough_data"] is False
    assert res["detail"]["segments"]["hotel:City Hotel"]["total"] == 10


def test_segment_accuracy_check_no_matured_labels(checks_env):
    from hotel_demo.agent_checks import segment_accuracy

    # Clear fixture labels to avoid conflicts
    for f in (data_dir() / "labels").glob("*.parquet"):
        f.unlink()

    # All bookings have arrival_date >= check_date, so no matured labels
    booking_rows = [{
        "booking_id": 30000,
        "arrival_date": "2026-02-20",  # after check date 2026-02-15
        "hotel": "City Hotel",
        "market_segment": "Direct",
        "is_canceled": 1,
        "lead_time": 30,
    }]
    pred_rows = [{
        "booking_id": 30000,
        "p_cancel": 0.9,
    }]

    _write_segment_accuracy_case(
        check_date="2026-02-15",
        label_date="2026-02-10",
        booking_rows=booking_rows,
        pred_rows=pred_rows,
    )

    res = segment_accuracy.run("2026-02-15")

    assert res["fired"] is False
    assert res["value"] is None
    assert res["detail"] == "no matured labels in window"


def test_segment_accuracy_check_runs_twice_appends(checks_env):
    from hotel_demo.agent_checks import segment_accuracy

    start1 = datetime.datetime.now(datetime.UTC)
    res1 = segment_accuracy.run("2026-01-13")
    end1 = datetime.datetime.now(datetime.UTC)

    start2 = datetime.datetime.now(datetime.UTC)
    res2 = segment_accuracy.run("2026-01-13")
    end2 = datetime.datetime.now(datetime.UTC)

    lines = _read_agent_result_lines("2026-01-13")
    assert len(lines) == 2
    assert lines[0] == res1
    assert lines[1] == res2
    _assert_utc_iso_timestamp(lines[0]["ran_at"], start1, end1)
    _assert_utc_iso_timestamp(lines[1]["ran_at"], start2, end2)
    assert lines[0]["ran_at"] <= lines[1]["ran_at"]


def test_categorical_drift_check_quiet(checks_env):
    from hotel_demo.agent_checks import categorical_drift

    assert categorical_drift.CHECK_NAME == "categorical_drift"
    assert categorical_drift.THRESHOLD == 0.2

    # Fixtures have bookings from 2025-12-01 to 2026-01-15.
    # Training cutoff is 2016-07-01, so no fixture data is before cutoff.
    # Use a date where we can see the "no training data" case works.
    start = datetime.datetime.now(datetime.UTC)
    res = categorical_drift.run("2026-01-15")
    end = datetime.datetime.now(datetime.UTC)

    assert set(res.keys()) == {"check", "ran_at", "fired", "value", "threshold", "detail"}
    assert res["check"] == "categorical_drift"
    assert res["threshold"] == 0.2
    assert res["fired"] is False
    assert res["value"] is None
    assert res["detail"] == "no training data"
    _assert_utc_iso_timestamp(res["ran_at"], start, end)

    lines = _read_agent_result_lines("2026-01-15")
    assert len(lines) == 1
    assert lines[0] == res


def test_categorical_drift_check_fires(checks_env):
    from hotel_demo.agent_checks import categorical_drift

    # Training cutoff is 2016-07-01, and fixtures don't have data before that.
    # To test drift firing, we need training data before the cutoff.
    # We can add training data by writing a file with booking_date before 2016-07-01.
    d = data_dir()

    # Clear all bookings
    for f in (d / "bookings").glob("*.parquet"):
        f.unlink()

    # Add training data: 50% City Hotel, 50% Resort Hotel, booking_date before 2016-07-01
    training_rows = []
    for i in range(50):
        hotel = "City Hotel" if i % 2 == 0 else "Resort Hotel"
        training_rows.append({
            "booking_id": 50000 + i,
            "booking_date": "2016-01-01",
            "arrival_date": "2016-02-01",
            "hotel": hotel,
            "market_segment": "Direct",
            "lead_time": 30,
            "arrival_month": 2,
            "arrival_week": 5,
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 3,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "PRT",
            "distribution_channel": "Direct",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "booking_changes": 0,
            "deposit_type": "No Deposit",
            "agent": None,
            "company": None,
            "customer_type": "Transient",
            "adr": 100.0,
            "total_of_special_requests": 0,
        })
    pd.DataFrame(training_rows).to_parquet(d / "bookings" / "2016-01-01.parquet", index=False)

    # Add serving data: 100% City Hotel, booking_date in the serving window
    serving_rows = []
    for i in range(50):
        serving_rows.append({
            "booking_id": 60000 + i,
            "booking_date": "2026-01-10",
            "arrival_date": "2026-02-10",
            "hotel": "City Hotel",  # All City Hotel - drift!
            "market_segment": "Direct",
            "lead_time": 30,
            "arrival_month": 2,
            "arrival_week": 2,
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 3,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "PRT",
            "distribution_channel": "Direct",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "booking_changes": 0,
            "deposit_type": "No Deposit",
            "agent": None,
            "company": None,
            "customer_type": "Transient",
            "adr": 100.0,
            "total_of_special_requests": 0,
        })
    pd.DataFrame(serving_rows).to_parquet(d / "bookings" / "2026-01-10.parquet", index=False)

    start = datetime.datetime.now(datetime.UTC)
    res = categorical_drift.run("2026-01-15")
    end = datetime.datetime.now(datetime.UTC)

    assert res["fired"] is True
    assert res["value"] > 0.2
    assert "worst_feature" in res["detail"]
    assert res["detail"]["worst_feature"] == "hotel"
    _assert_utc_iso_timestamp(res["ran_at"], start, end)


def test_categorical_drift_check_not_enough_data(checks_env):
    from hotel_demo.agent_checks import categorical_drift

    # Clear all bookings
    for f in (data_dir() / "bookings").glob("*.parquet"):
        f.unlink()

    # No training data
    res = categorical_drift.run("2026-01-15")

    assert res["fired"] is False
    assert res["value"] is None
    assert res["detail"] == "no training data"


def test_write_agent_result_line_stamps_ran_at_and_appends(checks_env):
    from hotel_demo.monitoring import write_agent_result_line

    record = {
        "check": "custom_agent_check",
        "fired": False,
        "value": 0.5,
        "threshold": 1.0,
        "detail": {"info": "test"},
    }
    start = datetime.datetime.now(datetime.UTC)
    res = write_agent_result_line("2026-01-11", record)
    end = datetime.datetime.now(datetime.UTC)

    assert res["check"] == "custom_agent_check"
    assert res["fired"] is False
    assert res["value"] == 0.5
    assert res["threshold"] == 1.0
    assert res["detail"] == {"info": "test"}
    _assert_utc_iso_timestamp(res["ran_at"], start, end)

    lines = _read_agent_result_lines("2026-01-11")
    assert len(lines) == 1
    assert lines[0] == res