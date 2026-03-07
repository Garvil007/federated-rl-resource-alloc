# src/mlops/model_registry.py

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
import json


class ModelRegistry:
    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_federated_model(
        self,
        model,
        config: dict,
        metrics: dict,
        run_name: str,
        model_name: str = "fed-rl-resource-alloc",
    ):
        """Register a trained federated RL model."""
        with mlflow.start_run(run_name=run_name) as run:
            # Log all hyperparameters
            mlflow.log_params({
                "num_clients": config["num_clients"],
                "num_rounds": config["num_rounds"],
                "local_epochs": config["local_epochs"],
                "strategy": config["strategy"],
                "learning_rate": config["training"]["lr"],
                "mu": config.get("mu", "N/A"),
            })

            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model artifact
            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name=model_name,
            )

            # Log configs as artifacts
            mlflow.log_dict(config, "federation_config.json")

            # Log tags
            mlflow.set_tags({
                "strategy": config["strategy"],
                "num_clients": str(config["num_clients"]),
                "environment": "resource_allocation",
            })

            return run.info.run_id

    def promote_model(self, model_name: str, version: int, stage: str):
        """Promote model to Staging/Production/Archived."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )
        print(f"Model {model_name} v{version} promoted to {stage}")

    def get_production_model(self, model_name: str):
        """Load the current production model."""
        model_uri = f"models:/{model_name}/Production"
        return mlflow.pytorch.load_model(model_uri)

    def compare_models(self, model_name: str) -> list:
        """Get all versions for comparison."""
        versions = self.client.search_model_versions(f"name='{model_name}'")
        return [
            {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "status": v.status,
            }
            for v in versions
        ]