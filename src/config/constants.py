"""
Central configuration constants for the Amazon reviews classification project.
"""

from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SERVING_ARTIFACTS_DIR = PROJECT_ROOT / "serving_artifacts"


# Dataset configuration
DATASET_FILENAME = "Amazon_initial_50_30_10000.arff"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
TARGET_COLUMN = "class"
EXPECTED_ROWS = 1480
EXPECTED_FEATURES = 9999
EXPECTED_AUTHORS = 50


# Train / test split
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY_SPLIT = True


# Evaluation
PRIMARY_METRIC = "f1_macro"
SECONDARY_METRIC = "accuracy"


# Optuna configuration
OPTUNA_TRIALS = 2
OPTUNA_CV_FOLDS = 2


# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "amazon-reviews-authorship"
MLFLOW_DEFAULT_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
REGISTERED_MODEL_NAME = "amazon-reviews-author-classifier"


# Model configuration
MODEL_NAMES = [
    "MultinomialNB",
    "LogisticRegression",
    "LinearSVC",
]

OPTIMIZED_MODEL_NAMES = [
    "LogisticRegression",
]

LOGISTIC_REGRESSION_CONFIG = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

MULTINOMIAL_NB_CONFIG = {
    "alpha": 1.0,
}

LINEAR_SVC_CONFIG = {
    "random_state": RANDOM_STATE,
    "dual": "auto",
    "max_iter": 5000,
}


# Prefect configuration
PREFECT_FLOW_NAME = "Amazon Reviews Classification Pipeline"
PREFECT_DEPLOYMENT_NAME = "amazon-reviews-training"


# Ensure local directories exist when needed by training / tracking code
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
SERVING_ARTIFACTS_DIR.mkdir(exist_ok=True)
