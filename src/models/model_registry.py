"""
Helpers to register the selected best model in MLflow Model Registry.
"""

import json
import logging
import shutil
from pathlib import Path

import mlflow
from mlflow.artifacts import download_artifacts

from src.config import SERVING_ARTIFACTS_DIR


logger = logging.getLogger(__name__)


def register_best_model(run_id: str, model_name: str) -> str:
    """
    Register a trained model artifact from an MLflow run.

    Args:
        run_id: MLflow run ID containing the logged model artifact.
        model_name: Target registered model name in MLflow.

    Returns:
        Registered model version as a string.
    """
    model_uri = f"runs:/{run_id}/model"
    logger.info("Registering model from %s into MLflow Model Registry as %s", model_uri, model_name)

    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(registered_model.version)
    logger.info("Model registered successfully as version %s", version)
    return version


def export_best_model_for_serving(
    run_id: str,
    model_name: str,
    metrics: dict | None = None,
    destination_dir: Path | None = None,
) -> Path:
    """
    Export the best MLflow model to a local directory for serving.

    This creates a self-contained folder that Docker or a cloud service can
    package without requiring direct access to the MLflow tracking store.
    """
    destination_dir = destination_dir or SERVING_ARTIFACTS_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)

    model_output_dir = destination_dir / "best_model"
    if model_output_dir.exists():
        shutil.rmtree(model_output_dir)

    artifact_uri = f"runs:/{run_id}/model"
    downloaded_path = Path(download_artifacts(artifact_uri=artifact_uri, dst_path=str(destination_dir)))
    shutil.copytree(downloaded_path, model_output_dir)

    metadata = {
        "run_id": run_id,
        "model_name": model_name,
        "metrics": metrics or {},
    }
    metadata_path = destination_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Exported best model for serving to %s", model_output_dir)
    return model_output_dir
