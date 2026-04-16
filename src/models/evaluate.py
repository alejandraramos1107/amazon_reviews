"""
Evaluation utilities for Amazon reviews classification models.
"""

from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute standard multiclass classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def build_classification_report(y_true, y_pred) -> str:
    """Return a text classification report for analysis or artifact logging."""
    return classification_report(y_true, y_pred, zero_division=0)
