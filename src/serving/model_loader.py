"""
Model loading utilities for the Amazon reviews demo interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import DATASET_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_DEFAULT_URI, TARGET_COLUMN
from src.data.loaders import load_amazon_reviews
from src.features.engineering import FeatureEngineer


logger = logging.getLogger(__name__)


@dataclass
class LoadedReviewModel:
    model: object
    model_name: str
    run_id: str
    metrics: dict
    dataset: pd.DataFrame
    transformed_dataset: object | None
    feature_columns: list[str]


class ReviewModelLoader:
    """Load the best MLflow model and the matching inference data context."""

    def __init__(self) -> None:
        self._loaded: LoadedReviewModel | None = None

    def load(self) -> LoadedReviewModel:
        if self._loaded is not None:
            return self._loaded

        mlflow.set_tracking_uri(MLFLOW_DEFAULT_URI)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            raise RuntimeError(
                f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' was not found. "
                "Run the training pipeline first."
            )

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED' and params.model_name != ''",
            order_by=["metrics.f1_macro DESC"],
            max_results=20,
        )
        if not runs:
            raise RuntimeError("No finished training runs were found in MLflow.")

        best_run = None
        for run in runs:
            if "model" in [artifact.path for artifact in client.list_artifacts(run.info.run_id)]:
                best_run = run
                break

        if best_run is None:
            raise RuntimeError("No MLflow run with a logged model artifact was found.")

        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        model_name = best_run.data.params.get("model_name", "unknown")

        dataset = load_amazon_reviews(str(DATASET_PATH))
        feature_columns = [column for column in dataset.columns if column != TARGET_COLUMN]
        features = dataset[feature_columns]

        transformed_dataset = None
        if model_name in {"LogisticRegression", "LinearSVC"}:
            feature_engineer = FeatureEngineer(target_col=TARGET_COLUMN)
            feature_engineer.build_pipeline(dataset)
            transformed_dataset = feature_engineer.apply_transform(features, fit=True)

        self._loaded = LoadedReviewModel(
            model=model,
            model_name=model_name,
            run_id=run_id,
            metrics=best_run.data.metrics,
            dataset=dataset,
            transformed_dataset=transformed_dataset,
            feature_columns=feature_columns,
        )
        logger.info("Loaded best MLflow model: %s (%s)", model_name, run_id)
        return self._loaded

    def predict_by_index(self, row_index: int) -> dict:
        loaded = self.load()
        if row_index < 0 or row_index >= len(loaded.dataset):
            raise IndexError(f"Row index out of range. Valid range: 0 to {len(loaded.dataset) - 1}.")

        row = loaded.dataset.iloc[row_index]
        actual_author = row[TARGET_COLUMN]

        if loaded.model_name == "MultinomialNB":
            model_input = loaded.dataset.iloc[[row_index]][loaded.feature_columns]
        else:
            if loaded.transformed_dataset is None:
                raise RuntimeError("Transformed dataset is not available for this model.")
            model_input = loaded.transformed_dataset[row_index : row_index + 1]

        predicted_author = loaded.model.predict(model_input)[0]
        return {
            "row_index": row_index,
            "predicted_author": str(predicted_author),
            "actual_author": str(actual_author),
            "correct": str(predicted_author) == str(actual_author),
            "model_name": loaded.model_name,
            "run_id": loaded.run_id,
            "f1_macro": loaded.metrics.get("f1_macro"),
            "accuracy": loaded.metrics.get("accuracy"),
        }


model_loader = ReviewModelLoader()
