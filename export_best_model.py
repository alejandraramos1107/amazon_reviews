#!/usr/bin/env python
"""
Export the best finished MLflow model into a local serving_artifacts folder.
"""

import mlflow
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_DEFAULT_URI, MLFLOW_EXPERIMENT_NAME
from src.models.model_registry import export_best_model_for_serving


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_DEFAULT_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED' and params.model_name != ''",
        order_by=["metrics.f1_macro DESC"],
        max_results=20,
    )
    if not runs:
        raise RuntimeError("No finished model runs found.")

    best_run = None
    for run in runs:
        if "model" in [artifact.path for artifact in client.list_artifacts(run.info.run_id)]:
            best_run = run
            break

    if best_run is None:
        raise RuntimeError("No MLflow run with a logged model artifact was found.")

    export_best_model_for_serving(
        run_id=best_run.info.run_id,
        model_name=best_run.data.params.get("model_name", "unknown"),
        metrics=best_run.data.metrics,
    )
    print("Best model exported to serving_artifacts/.")


if __name__ == "__main__":
    main()
