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
    final_weights = ray.get(server.run_federation.remote())

    # Save final global model
    import torch
    torch.save(final_weights, "outputs/best_model.pt")
    wandb.save("outputs/best_model.pt")

    wandb.finish()
    ray.shutdown()
    print("Federated training complete!")


if __name__ == "__main__":
    main()