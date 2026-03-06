import ray
import torch
from src.envs.resource_alloc_env import ResourceAllocationEnv
from src.agents.policy_network import PolicyNetwork


@ray.remote
class FederatedClient:
    def __init__(self, client_id: int, env_config: dict, train_config: dict):
        self.client_id = client_id
        self.env = ResourceAllocationEnv(config=env_config)
        self.train_config = train_config

        obs_dim = train_config["obs_dim"]
        act_dim = train_config["act_dim"]
        self.policy = PolicyNetwork(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=train_config.get("lr", 3e-4),
        )
        self.global_weights = None  # Store for FedProx

    def set_weights(self, global_weights):
        """Receive global model weights from server."""
        self.policy.load_state_dict(global_weights)
        self.global_weights = {k: v.clone() for k, v in global_weights.items()}

    def train_local(self, num_epochs: int):
        """Train locally for E epochs, return updated weights."""
        total_samples = 0
        total_reward = 0.0
        total_loss = 0.0
        total_sla = 0.0
        mu = self.train_config.get("mu", 0.01)
        use_prox = self.train_config.get("strategy", "fedavg") == "fedprox"

        for epoch in range(num_epochs):
            # Collect rollout
            obs, _ = self.env.reset()
            episode_reward = 0.0
            episode_samples = 0

            for step in range(self.env.episode_length):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action_logits = self.policy(obs_tensor)
                    action = self._sample_action(action_logits)

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_samples += 1
                obs = next_obs

                if terminated or truncated:
                    break

            # Compute policy gradient loss (simplified PPO)
            loss = self._compute_loss()

            # Add FedProx proximal term if applicable
            if use_prox and self.global_weights is not None:
                prox_term = 0.0
                for key in self.policy.state_dict():
                    local_w = self.policy.state_dict()[key]
                    global_w = self.global_weights[key]
                    prox_term += torch.norm(local_w - global_w) ** 2
                loss += (mu / 2) * prox_term

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_samples += episode_samples
            total_reward += episode_reward
            total_loss += loss.item()
            total_sla += info.get("sla_rate", 0.0)

        metrics = {
            "reward": total_reward / num_epochs,
            "loss": total_loss / num_epochs,
            "sla_rate": total_sla / num_epochs,
            "samples": total_samples,
        }

        return (
            self.policy.state_dict(),
            total_samples,
            metrics,
        )

    def _sample_action(self, logits):
        """Sample action from policy logits."""
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().numpy().flatten()

    def _compute_loss(self):
        """Simplified policy gradient loss. Replace with full PPO in production."""
        # Placeholder — in real implementation, use collected trajectory buffer
        return torch.tensor(0.0, requires_grad=True)
