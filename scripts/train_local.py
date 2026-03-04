import ray
from ray import tune
from ray.air import RunConfig, CheckpointConfig
from src.agents.ppo_agent import build_ppo_config
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="training/ppo_default")
def train(cfg: DictConfig):
    ray.init()
    ppo_config = build_ppo_config(cfg.env, cfg.training)

    tuner = tune.Tuner(
        "PPO",
        param_space=ppo_config.to_dict(),
        run_config=RunConfig(
            name="ppo_resource_alloc_baseline",
            stop={"training_iteration": cfg.training.max_iters},
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
                num_to_keep=3,
            ),
            storage_path="./ray_results",
        ),
    )
    results = tuner.fit()
    best = results.get_best_result("episode_reward_mean", "max")
    print(f"Best reward: {best.metrics['episode_reward_mean']}")
    ray.shutdown()


if __name__ == "__main__":
    train()
