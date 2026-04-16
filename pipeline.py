#!/usr/bin/env python
"""
Amazon reviews classification pipeline orchestrated with Prefect.
"""

import logging
from typing import Tuple

import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    MLFLOW_EXPERIMENT_NAME,
    OPTIMIZED_MODEL_NAMES,
    PREFECT_FLOW_NAME,
    PRIMARY_METRIC,
    RANDOM_STATE,
    SECONDARY_METRIC,
    STRATIFY_SPLIT,
    TARGET_COLUMN,
    TEST_SIZE,
    setup_mlflow,
)
from src.data.loaders import load_amazon_reviews
from src.data.validators import run_all_validations
from src.features.engineering import FeatureEngineer
from src.models.trainer import (
    optimize_model_hyperparameters,
    select_best_model,
    train_and_evaluate_models,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(name="load_data", description="Load Amazon reviews dataset")
def load_data_task(dataset_path: str) -> pd.DataFrame:
    """Load the dataset from disk."""
    logger.info("Loading dataset from %s", dataset_path)
    return load_amazon_reviews(dataset_path)


@task(name="validate_data", description="Run schema and data quality validations")
def validate_data_task(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the dataset before splitting and training."""
    logger.info("Running dataset validations for dataframe with shape=%s", df.shape)
    is_valid = run_all_validations(df)
    if not is_valid:
        raise ValueError("Dataset validation failed.")
    logger.info("Dataset validations completed successfully")
    return df


@task(name="split_data", description="Split dataset into train and test partitions")
def split_data_task(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and testing partitions."""
    logger.info("Splitting dataset into train and test sets")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    stratify = y if STRATIFY_SPLIT else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    logger.info(
        "Split complete. X_train=%s, X_test=%s, y_train=%s, y_test=%s",
        X_train.shape,
        X_test.shape,
        y_train.shape,
        y_test.shape,
    )
    return X_train, X_test, y_train, y_test


@task(name="feature_engineering", description="Build and apply the feature engineering pipeline")
def feature_engineering_task(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[object, object]:
    """Fit feature engineering on train and transform both train and test."""
    logger.info("Starting feature engineering")
    feature_engineer = FeatureEngineer(target_col=TARGET_COLUMN)
    feature_engineer.build_pipeline(pd.concat([X_train, X_test], axis=0))

    X_train_transformed = feature_engineer.apply_transform(X_train, fit=True)
    X_test_transformed = feature_engineer.apply_transform(X_test, fit=False)
    logger.info(
        "Feature engineering complete. Transformed shapes: X_train=%s, X_test=%s",
        getattr(X_train_transformed, "shape", None),
        getattr(X_test_transformed, "shape", None),
    )
    return X_train_transformed, X_test_transformed


@flow(name=PREFECT_FLOW_NAME, log_prints=True)
def amazon_reviews_flow(dataset_path: str = str(DATASET_PATH)) -> str:
    """
    Main Prefect flow for Amazon reviews authorship classification.

    Returns:
        MLflow run ID associated with the selected best model.
    """
    tracking_uri = setup_mlflow()
    prefect_logger = get_run_logger()
    prefect_logger.info("Starting Amazon reviews pipeline")
    prefect_logger.info("MLflow tracking URI: %s", tracking_uri)
    prefect_logger.info("Dataset path: %s", dataset_path)

    prefect_logger.info("Step 1/6: Loading data")
    df = load_data_task(dataset_path)
    prefect_logger.info("Step 2/6: Validating data")
    df = validate_data_task(df)
    prefect_logger.info("Step 3/6: Splitting data")
    X_train, X_test, y_train, y_test = split_data_task(df)
    prefect_logger.info("Step 4/6: Applying feature engineering")
    X_train_transformed, X_test_transformed = feature_engineering_task(X_train, X_test)
    prefect_logger.info("Step 5/6: Running Optuna hyperparameter tuning")
    optimized_configs = optimize_model_hyperparameters(
        X_train_raw=X_train,
        y_train=y_train,
        X_train_transformed=X_train_transformed,
    )
    prefect_logger.info("Optuna finished. Best configs: %s", optimized_configs)

    prefect_logger.info("Step 6/6: Training final candidate models")
    results = train_and_evaluate_models(
        X_train_raw=X_train,
        X_test_raw=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_transformed=X_train_transformed,
        X_test_transformed=X_test_transformed,
        optimized_configs=optimized_configs,
    )
    best_result = select_best_model(results)

    prefect_logger.info(
        "Selected best model: %s with %s=%.4f",
        best_result["model_name"],
        PRIMARY_METRIC,
        best_result["metrics"][PRIMARY_METRIC],
    )
    prefect_logger.info("Creating final Prefect artifact summary")

    results_table = "\n".join(
        [
            (
                f"- **{result['model_name']}**: "
                f"{PRIMARY_METRIC}={result['metrics'][PRIMARY_METRIC]:.4f}, "
                f"{SECONDARY_METRIC}={result['metrics'][SECONDARY_METRIC]:.4f}, "
                f"run_id={result['run_id']}"
            )
            for result in results
        ]
    )

    pipeline_summary = f"""
    # Amazon Reviews Pipeline Summary

    ## Dataset
    - **Path**: `{dataset_path}`
    - **Rows**: {len(df):,}
    - **Features**: {X_train.shape[1]:,}
    - **Target column**: `{TARGET_COLUMN}`

    ## MLflow
    - **Tracking URI**: `{tracking_uri}`
    - **Experiment**: `{MLFLOW_EXPERIMENT_NAME}`

    ## Model Comparison
    {results_table}

    ## Hyperparameter Tuning
    - Optuna was used to tune these models before final training: {", ".join(OPTIMIZED_MODEL_NAMES)}

    ## Selected Candidate
    - **Model**: {best_result["model_name"]}
    - **Primary metric ({PRIMARY_METRIC})**: {best_result["metrics"][PRIMARY_METRIC]:.4f}
    - **Secondary metric ({SECONDARY_METRIC})**: {best_result["metrics"][SECONDARY_METRIC]:.4f}
    - **MLflow run ID**: `{best_result["run_id"]}`
    - **Best params**: `{best_result["params"]}`
    """

    create_markdown_artifact(
        key="amazon-reviews-pipeline-summary",
        markdown=pipeline_summary,
        description="Training summary and candidate model selection",
    )

    return best_result["run_id"]


if __name__ == "__main__":
    run_id = amazon_reviews_flow()
    print(f"Pipeline completed successfully. Best model run_id: {run_id}")
