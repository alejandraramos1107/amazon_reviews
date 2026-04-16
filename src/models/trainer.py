"""
Training and Optuna optimization utilities for the Amazon reviews project.
"""

import logging
import time
from typing import Any, Dict, List

import mlflow
import mlflow.sklearn
import optuna
from prefect import get_run_logger, task
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import (
    LINEAR_SVC_CONFIG,
    LOGISTIC_REGRESSION_CONFIG,
    MODEL_NAMES,
    MULTINOMIAL_NB_CONFIG,
    OPTUNA_CV_FOLDS,
    OPTIMIZED_MODEL_NAMES,
    OPTUNA_TRIALS,
    PRIMARY_METRIC,
    RANDOM_STATE,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute evaluation metrics for multiclass classification."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _build_model(model_name: str, params: Dict[str, Any] | None = None) -> Any:
    """Instantiate a model by name, optionally overriding hyperparameters."""
    params = params or {}

    if model_name == "MultinomialNB":
        config = {**MULTINOMIAL_NB_CONFIG, **params}
        return MultinomialNB(**config)
    if model_name == "LogisticRegression":
        config = {**LOGISTIC_REGRESSION_CONFIG, **params}
        return LogisticRegression(**config)
    if model_name == "LinearSVC":
        config = {**LINEAR_SVC_CONFIG, **params}
        return LinearSVC(**config)

    raise ValueError(f"Unsupported model: {model_name}")


def _suggest_params(model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
    """Define the Optuna search space for each supported model."""
    if model_name == "MultinomialNB":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 3.0, log=True),
        }

    if model_name == "LogisticRegression":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    raise ValueError(f"Unsupported model for optimization: {model_name}")


def _select_feature_space(
    model_name: str,
    X_train_raw,
    X_train_transformed,
    X_test_raw,
    X_test_transformed,
):
    """Use raw counts for Naive Bayes and transformed features for linear models."""
    if model_name == "MultinomialNB":
        return X_train_raw, X_test_raw
    return X_train_transformed, X_test_transformed


@task(name="optimize_models", description="Tune model hyperparameters with Optuna")
def optimize_model_hyperparameters(
    X_train_raw,
    y_train,
    X_train_transformed=None,
) -> Dict[str, Dict[str, Any]]:
    """
    Optimize candidate model hyperparameters using Optuna and cross-validation.

    Returns:
        Dictionary keyed by model name with best params and best CV score.
    """
    prefect_logger = get_run_logger()
    transformed_train = X_train_transformed if X_train_transformed is not None else X_train_raw

    best_configs: Dict[str, Dict[str, Any]] = {}
    cv = StratifiedKFold(
        n_splits=OPTUNA_CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for model_name in OPTIMIZED_MODEL_NAMES:
        current_X_train, _ = _select_feature_space(
            model_name=model_name,
            X_train_raw=X_train_raw,
            X_train_transformed=transformed_train,
            X_test_raw=None,
            X_test_transformed=None,
        )

        prefect_logger.info(
            "Optimizing hyperparameters for %s with %s trials",
            model_name,
            OPTUNA_TRIALS,
        )

        def objective(trial: optuna.Trial) -> float:
            trial_params = _suggest_params(model_name, trial)
            model = _build_model(model_name, trial_params)
            prefect_logger.info(
                "Starting Optuna trial %s/%s for %s with params=%s",
                trial.number + 1,
                OPTUNA_TRIALS,
                model_name,
                trial_params,
            )

            with mlflow.start_run(run_name=f"{model_name}_trial_{trial.number}", nested=True):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("optimization_framework", "optuna")
                mlflow.log_params(trial_params)

                scores = cross_val_score(
                    model,
                    current_X_train,
                    y_train,
                    cv=cv,
                    scoring=PRIMARY_METRIC,
                    n_jobs=None,
                )
                mean_score = float(scores.mean())
                prefect_logger.info(
                    "Finished Optuna trial %s/%s for %s with cv_%s=%.4f",
                    trial.number + 1,
                    OPTUNA_TRIALS,
                    model_name,
                    PRIMARY_METRIC,
                    mean_score,
                )

                mlflow.log_metric("cv_mean_f1_macro", mean_score)
                mlflow.log_metric("cv_std_f1_macro", float(scores.std()))
                return mean_score

        study = optuna.create_study(direction="maximize", study_name=f"{model_name.lower()}-optimization")
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

        best_params = study.best_params
        best_score = float(study.best_value)
        best_configs[model_name] = {
            "params": best_params,
            "cv_best_score": best_score,
        }

        prefect_logger.info(
            "Best %s params found. %s=%.4f",
            model_name,
            PRIMARY_METRIC,
            best_score,
        )
        prefect_logger.info("Best params for %s: %s", model_name, best_params)

    return best_configs


@task(name="train_models", description="Train and compare classification models")
def train_and_evaluate_models(
    X_train_raw,
    X_test_raw,
    y_train,
    y_test,
    X_train_transformed=None,
    X_test_transformed=None,
    optimized_configs: Dict[str, Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """
    Train the configured candidate models and log results to MLflow.

    Returns:
        List of dictionaries with model metadata, metrics, and run IDs.
    """
    prefect_logger = get_run_logger()
    transformed_train = X_train_transformed if X_train_transformed is not None else X_train_raw
    transformed_test = X_test_transformed if X_test_transformed is not None else X_test_raw
    optimized_configs = optimized_configs or {}
    results: List[Dict[str, Any]] = []

    for model_name in MODEL_NAMES:
        tuned_params = optimized_configs.get(model_name, {}).get("params", {})
        model = _build_model(model_name, tuned_params)
        current_X_train, current_X_test = _select_feature_space(
            model_name=model_name,
            X_train_raw=X_train_raw,
            X_train_transformed=transformed_train,
            X_test_raw=X_test_raw,
            X_test_transformed=transformed_test,
        )

        prefect_logger.info("Training model: %s", model_name)
        prefect_logger.info(
            "Using feature space for %s: %s",
            model_name,
            "raw counts" if model_name == "MultinomialNB" else "scaled features",
        )
        prefect_logger.info("Training params for %s: %s", model_name, model.get_params())

        start_time = time.perf_counter()
        with mlflow.start_run(run_name=model_name):
            model.fit(current_X_train, y_train)
            prefect_logger.info("Model %s finished fit stage, generating predictions...", model_name)
            y_pred = model.predict(current_X_test)
            elapsed_time = time.perf_counter() - start_time

            metrics = _calculate_metrics(y_test, y_pred)
            metrics["training_time_seconds"] = elapsed_time

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("optimized_with_optuna", bool(tuned_params))
            mlflow.log_params(model.get_params())
            if model_name in optimized_configs:
                mlflow.log_metric("optuna_best_cv_f1_macro", optimized_configs[model_name]["cv_best_score"])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            run_id = mlflow.active_run().info.run_id
            prefect_logger.info(
                "Model %s finished. %s=%.4f, accuracy=%.4f, time=%.2fs",
                model_name,
                PRIMARY_METRIC,
                metrics[PRIMARY_METRIC],
                metrics["accuracy"],
                elapsed_time,
            )

        results.append(
            {
                "model_name": model_name,
                "metrics": metrics,
                "run_id": run_id,
                "params": model.get_params(),
            }
        )

    return results


@task(name="select_best_model", description="Select best model according to primary metric")
def select_best_model(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the best model based on the configured primary metric."""
    prefect_logger = get_run_logger()

    if not results:
        raise ValueError("No model results were provided for selection.")

    best_result = max(results, key=lambda result: result["metrics"][PRIMARY_METRIC])
    prefect_logger.info(
        "Best model: %s with %s=%.4f",
        best_result["model_name"],
        PRIMARY_METRIC,
        best_result["metrics"][PRIMARY_METRIC],
    )
    prefect_logger.info("Best model params: %s", best_result["params"])
    return best_result
