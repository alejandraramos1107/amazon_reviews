"""
Helpers to register the selected best model in MLflow Model Registry.
"""

import logging

import mlflow


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
