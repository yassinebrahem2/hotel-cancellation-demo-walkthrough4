"""Missing evaluation check - flags production model having no fresh evaluation."""

import datetime
import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from hotel_demo.features import MODEL_ALIAS, MODEL_NAME
from hotel_demo.paths import tracking_uri

CHECK_NAME = "missing_evaluation"
MAX_AGE_DAYS = 7  # evaluation older than this is considered stale


def run(date: str) -> dict[str, Any]:
    """Run the missing evaluation check for the given date.

    Checks if the production model has a fresh evaluation artifact.
    Fires if no evaluation exists or it's older than MAX_AGE_DAYS.
    """
    ran_at = datetime.datetime.now(datetime.UTC).isoformat()

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()

    try:
        # Get the production model version
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        version = mv.version
        run_id = mv.run_id
    except mlflow.exceptions.MlflowException:
        return {
            "check_id": CHECK_NAME,
            "origin": "agent",
            "ran_at": ran_at,
            "fired": True,
            "value": None,
            "threshold": MAX_AGE_DAYS,
            "category": "operational",
            "subject": f"model {MODEL_NAME}@{MODEL_ALIAS}",
            "model": "hotel-cancellation",
            "detail": {"reason": "no production alias", "version": None},
            "source_ref": "checks/missing_evaluation.py",
            "version": None,
            "not_enough_data": False,
        }

    # Check for evaluation artifact
    eval_artifact_path = "mlops_agents/evaluation.json"
    artifacts = client.list_artifacts(run_id, path=eval_artifact_path)

    if not artifacts:
        return {
            "check_id": CHECK_NAME,
            "origin": "agent",
            "ran_at": ran_at,
            "fired": True,
            "value": None,
            "threshold": MAX_AGE_DAYS,
            "category": "operational",
            "subject": f"model {MODEL_NAME}@{MODEL_ALIAS}",
            "model": "hotel-cancellation",
            "detail": {"reason": "no evaluation artifact", "version": version, "run_id": run_id},
            "source_ref": "checks/missing_evaluation.py",
            "version": None,
            "not_enough_data": False,
        }

    # Download and parse the evaluation artifact to check its timestamp
    try:
        local_path = client.download_artifacts(run_id, eval_artifact_path)
        with Path(local_path).open() as f:
            eval_data = json.load(f)

        eval_time_str = eval_data.get("evaluated_at") or eval_data.get("ran_at")
        if not eval_time_str:
            return {
                "check_id": CHECK_NAME,
                "origin": "agent",
                "ran_at": ran_at,
                "fired": True,
                "value": None,
                "threshold": MAX_AGE_DAYS,
                "category": "operational",
                "subject": f"model {MODEL_NAME}@{MODEL_ALIAS}",
                "model": "hotel-cancellation",
                "detail": {
                    "reason": "evaluation missing timestamp",
                    "version": version,
                    "run_id": run_id,
                },
                "source_ref": "checks/missing_evaluation.py",
                "version": None,
                "not_enough_data": False,
            }

        eval_time = datetime.datetime.fromisoformat(eval_time_str.replace("Z", "+00:00"))
        age_days = (datetime.datetime.now(datetime.UTC) - eval_time).days

        if age_days > MAX_AGE_DAYS:
            return {
                "check_id": CHECK_NAME,
                "origin": "agent",
                "ran_at": ran_at,
                "fired": True,
                "value": age_days,
                "threshold": MAX_AGE_DAYS,
                "category": "operational",
                "subject": f"model {MODEL_NAME}@{MODEL_ALIAS}",
                "model": "hotel-cancellation",
                "detail": {
                    "reason": "evaluation stale",
                    "age_days": age_days,
                    "version": version,
                    "evaluated_at": eval_time_str,
                },
                "source_ref": "checks/missing_evaluation.py",
                "version": None,
                "not_enough_data": False,
            }

        return {
            "check_id": CHECK_NAME,
            "origin": "agent",
            "ran_at": ran_at,
            "fired": False,
            "value": age_days,
            "threshold": MAX_AGE_DAYS,
            "category": "operational",
            "subject": f"model {MODEL_NAME}@{MODEL_ALIAS}",
            "model": "hotel-cancellation",
            "detail": {"age_days": age_days, "version": version, "evaluated_at": eval_time_str},
            "source_ref": "checks/missing_evaluation.py",
            "version": None,
            "not_enough_data": False,
        }

    except Exception as e:
        return {
            "check_id": CHECK_NAME,
            "origin": "agent",
            "ran_at": ran_at,
            "fired": True,
            "value": None,
            "threshold": MAX_AGE_DAYS,
            "category": "operational",
            "subject": f"model {MODEL_NAME}@{MODEL_ALIAS}",
            "model": "hotel-cancellation",
            "detail": {
                "reason": f"error reading evaluation: {e}",
                "version": version,
                "run_id": run_id,
            },
            "source_ref": "checks/missing_evaluation.py",
            "version": None,
            "not_enough_data": False,
        }
