#!/usr/bin/env python
"""
Deployment configuration for the Amazon reviews classification pipeline.
"""

from pipeline import amazon_reviews_flow
from src.config import DATASET_PATH, PREFECT_DEPLOYMENT_NAME


if __name__ == "__main__":
    print("\nStarting Prefect deployment for Amazon reviews classification...")
    print(f"   Deployment name: {PREFECT_DEPLOYMENT_NAME}")
    print(f"   Dataset path: {DATASET_PATH}")

    amazon_reviews_flow.serve(
        name=PREFECT_DEPLOYMENT_NAME,
        tags=["amazon-reviews", "classification", "ml", "training"],
        description="Train, optimize, and compare classification models for Amazon reviews authorship.",
        parameters={
            "dataset_path": str(DATASET_PATH),
        },
    )

    print("\nDeployment is now running.")
    print("Check executions in Prefect Cloud or your configured Prefect UI.")
