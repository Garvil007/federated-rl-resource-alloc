import wandb
from typing import Dict
import git


class ExperimentTracker:
    """Unified experiment tracking with W&B."""

    def __init__(self, project: str, config: Dict, tags: list = None):
        repo = git.Repo(search_parent_directories=True)
        self.run = wandb.init(
            project=project,
            config=config,
            tags=tags or [],
            notes=f"git_sha: {repo.head.object.hexsha[:8]}",
            group=config.get("experiment_group", "default"),
        )
        # Log git info for reproducibility
        wandb.config.update({
            "git_sha": repo.head.object.hexsha,
            "git_branch": repo.active_branch.name,
            "git_dirty": repo.is_dirty(),
        })

    def log_round(self, round_num: int, metrics: Dict):
        """Log federation round metrics."""
        wandb.log({"round": round_num, **metrics})

    def log_client(self, round_num: int, client_id: int, metrics: Dict):
        """Log per-client metrics for analysis."""
        prefixed = {f"client_{client_id}/{k}": v for k, v in metrics.items()}
        wandb.log({"round": round_num, **prefixed})

    def log_model(self, model_path: str, aliases: list = None):
        """Log trained model as W&B artifact."""
        artifact = wandb.Artifact(
            name="fed-rl-model",
            type="model",
            metadata=dict(wandb.config),
        )
        artifact.add_file(model_path)
        self.run.log_artifact(artifact, aliases=aliases or ["latest"])

    def log_env_config(self, client_configs: list):
        """Log environment configs as table for analysis."""
        table = wandb.Table(
            columns=["client_id", "num_nodes", "arrival_rate", "cpu_range"],
            data=[
                [i, c["num_nodes"], c["task_arrival_rate"],
                 str(c.get("cpu_range", "default"))]
                for i, c in enumerate(client_configs)
            ]
        )
        wandb.log({"client_configs": table})

    def log_comparison_table(self, results: Dict[str, Dict]):
        """Log comparison between strategies/baselines."""
        columns = ["method", "avg_reward", "sla_rate", "convergence_round"]
        data = [[k, v["reward"], v["sla_rate"], v["convergence"]]
                for k, v in results.items()]
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"strategy_comparison": table})

    def finish(self):
        wandb.finish()