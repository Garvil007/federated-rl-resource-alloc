import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from src.federation.server import FederationServer


@hydra.main(version_base=None, config_path="../configs/training", config_name="federated_5clients")
def main(cfg: DictConfig):
    # Initialize Ray
    ray.init()

    # Initialize W&B
    wandb.init(
        project="federated-rl-resource-alloc",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["federated", cfg.federation.strategy],
    )

    # Convert config
    config = OmegaConf.to_container(cfg, resolve=True)

    # Create and run federation server
    server = FederationServer.remote(config)
    final_weights, log_history = ray.get(server.run_federation.remote())

    # Log metrics per round so WandB shows real-time charts
    for round_metrics in log_history:
        wandb.log(round_metrics, step=round_metrics["round"])

    # Log summary metrics for sweep comparison
    if log_history:
        last = log_history[-1]
        wandb.summary["final/avg_reward"] = last.get("global/avg_reward", 0)
        wandb.summary["final/weight_divergence"] = last.get("global/weight_divergence", 0)
        wandb.summary["final/reward_std"] = last.get("global/reward_std", 0)

        # Compute improvement trend (last 5 rounds vs first 5)
        if len(log_history) >= 10:
            early = sum(h.get("global/avg_reward", 0) for h in log_history[:5]) / 5
            late = sum(h.get("global/avg_reward", 0) for h in log_history[-5:]) / 5
            wandb.summary["final/reward_improvement"] = late - early

    # Save final global model
    import torch
    os.makedirs("outputs", exist_ok=True)
    torch.save(final_weights, "outputs/best_model.pt")
    wandb.save("outputs/best_model.pt")

    wandb.finish()
    ray.shutdown()
    print("Federated training complete!")


if __name__ == "__main__":
    main()