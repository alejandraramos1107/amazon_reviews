"""
Model loading utilities for the Amazon reviews demo interface.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from src.config import DATASET_PATH, SERVING_ARTIFACTS_DIR, TARGET_COLUMN
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

        metadata_path = SERVING_ARTIFACTS_DIR / "metadata.json"
        model_dir = SERVING_ARTIFACTS_DIR / "best_model"
        if not metadata_path.exists() or not model_dir.exists():
            raise RuntimeError(
                "Serving artifacts were not found. Run the training pipeline or "
                "`python export_best_model.py` first."
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_id = metadata.get("run_id", "unknown")
        model_name = metadata.get("model_name", "unknown")
        model = mlflow.sklearn.load_model(str(model_dir))

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
            metrics=metadata.get("metrics", {}),
            dataset=dataset,
            transformed_dataset=transformed_dataset,
            feature_columns=feature_columns,
        )
        logger.info("Loaded serving model: %s (%s) from %s", model_name, run_id, model_dir)
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
