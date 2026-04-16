"""
MLflow configuration helpers for the Amazon reviews project.
"""

import logging
import os

import mlflow

from .constants import MLFLOW_DEFAULT_URI, MLFLOW_EXPERIMENT_NAME


logger = logging.getLogger(__name__)


def setup_mlflow() -> str:
    """
    Configure MLflow for local development or external tracking.

    Priority:
    1. `MLFLOW_TRACKING_URI` environment variable
    2. Local SQLite fallback defined in constants

    Returns:
        Active MLflow tracking URI.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_DEFAULT_URI)

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.search_experiments()
        logger.info("Connected to MLflow at %s", tracking_uri)
    except Exception as exc:
        logger.warning(
            "Could not connect to MLflow at %s: %s. Falling back to local tracking.",
            tracking_uri,
            exc,
        )
        tracking_uri = MLFLOW_DEFAULT_URI
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("Using MLflow experiment: %s", MLFLOW_EXPERIMENT_NAME)

    return tracking_uri
