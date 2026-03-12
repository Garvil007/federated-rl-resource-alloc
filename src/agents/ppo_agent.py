from ray.rllib.algorithms.ppo import PPOConfig  # type: ignore[import-untyped]
from ray.tune.registry import register_env  # type: ignore[import-untyped]
from src.envs.resource_alloc_env import ResourceAllocationEnv  # type: ignore[import-untyped]


def env_creator(env_config: dict) -> ResourceAllocationEnv:
    return ResourceAllocationEnv(config=env_config)


register_env("ResourceAlloc-v0", env_creator)


def build_ppo_config(env_cfg: dict, train_cfg: dict) -> PPOConfig:
    config = (
        PPOConfig()
        .environment(
            env="ResourceAlloc-v0",
            env_config=env_cfg,
        )
        .framework("torch")
        .training(
            lr=train_cfg.get("lr", 3e-4),
            gamma=train_cfg.get("gamma", 0.99),
            lambda_=train_cfg.get("gae_lambda", 0.95),
            clip_param=train_cfg.get("clip_param", 0.2),
            num_sgd_iter=train_cfg.get("num_sgd_iter", 10),
            sgd_minibatch_size=train_cfg.get("minibatch", 256),
            train_batch_size=train_cfg.get("batch_size", 4000),
            entropy_coeff=train_cfg.get("entropy_coeff", 0.01),
            vf_loss_coeff=train_cfg.get("vf_loss_coeff", 0.5),
            model={
                "fcnet_hiddens": [256, 256, 128],
                "fcnet_activation": "relu",
            },
        )
        .rollouts(
            num_rollout_workers=train_cfg.get("workers", 4),
            rollout_fragment_length="auto",
        )
        .resources(
            num_gpus=train_cfg.get("gpus", 0),
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_num_workers=1,
            evaluation_duration=10,
        )
    )
    return config
