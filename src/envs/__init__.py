from gymnasium.envs.registration import register

register(
    id="ResourceAllocation-v0",
    entry_point="src.envs.resource_alloc_env:ResourceAllocationEnv",
    max_episode_steps=1000,
)
