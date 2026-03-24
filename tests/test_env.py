import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env
import src.envs  # registers the environment

def test_env_gymnasium_compliance():
    """Verify env passes Gymnasium API checker."""
    env = gym.make("ResourceAllocation-v0")
    check_env(env.unwrapped, skip_render_check=True)


def test_env_reset_returns_valid_obs():
    env = gym.make("ResourceAllocation-v0")
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_env_step_reward_range():
    env = gym.make("ResourceAllocation-v0")
    env.reset()
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    assert isinstance(reward, (int, float))


def test_env_episode_terminates():
    env = gym.make("ResourceAllocation-v0")
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == 1000


def test_env_heterogeneous_configs():
    """Test that different configs produce different envs."""
    cfg1 = {"num_nodes": 3, "task_arrival_rate": 2.0}
    cfg2 = {"num_nodes": 8, "task_arrival_rate": 5.0}
    env1 = gym.make("ResourceAllocation-v0", config=cfg1)
    env2 = gym.make("ResourceAllocation-v0", config=cfg2)
    assert env1.num_nodes != env2.num_nodes


def test_env_deterministic_with_seed():
    """Same seed should produce same trajectory."""
    env1 = gym.make("ResourceAllocation-v0")
    env2 = gym.make("ResourceAllocation-v0")
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)
