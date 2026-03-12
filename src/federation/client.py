import ray
import torch
import torch.nn.functional as F
import numpy as np
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
        hidden_dim = train_config.get("hidden_dim", 256)
        self.gamma = train_config.get("gamma", 0.99)
        self.entropy_coeff = train_config.get("entropy_coeff", 0.01)
        self.value_coeff = train_config.get("value_coeff", 0.5)
        self.max_nodes = self.env.max_nodes
        self.max_pending = self.env.max_pending

        self.policy = PolicyNetwork(obs_dim, act_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=train_config.get("lr", 3e-4),
        )
        # Cosine annealing for stable convergence
        total_steps = (
            train_config.get("num_rounds", 100)
            * train_config.get("local_epochs", 5)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-5
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
            # Collect rollout with trajectory buffer
            trajectory = self._collect_rollout()

            episode_reward = sum(trajectory["rewards"])
            episode_samples = len(trajectory["rewards"])

            # Compute loss from the collected trajectory
            loss = self._compute_loss(trajectory)

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
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_samples += episode_samples
            total_reward += episode_reward
            total_loss += loss.item()
            total_sla += trajectory["final_info"].get("sla_rate", 0.0)

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

    def _collect_rollout(self):
        """Run one episode, collecting full trajectory for training."""
        obs, _ = self.env.reset()
        trajectory = {
            "observations": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "final_info": {},
        }

        for step in range(self.env.episode_length):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            logits, value = self.policy(obs_tensor)

            # Sample action for MultiDiscrete: one decision per pending task
            action, log_prob = self._sample_action(logits)

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            trajectory["observations"].append(obs_tensor.squeeze(0))
            trajectory["actions"].append(torch.tensor(action, dtype=torch.long))
            trajectory["log_probs"].append(log_prob)
            trajectory["rewards"].append(reward)
            trajectory["values"].append(value.squeeze())

            obs = next_obs
            if terminated or truncated:
                trajectory["final_info"] = info
                break

        if not trajectory["final_info"]:
            trajectory["final_info"] = info

        return trajectory

    def _sample_action(self, logits):
        """Sample action from policy logits for MultiDiscrete action space.

        The policy outputs (max_nodes+1) logits for each of max_pending task slots.
        We reshape and sample independently per task slot.
        """
        num_slots = self.max_pending
        act_per_slot = self.max_nodes + 1

        # Reshape logits: [1, num_slots * act_per_slot] -> [num_slots, act_per_slot]
        logits_2d = logits.view(num_slots, act_per_slot)

        probs = torch.softmax(logits_2d, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()  # [num_slots]

        # Sum of log probs across all task slots
        log_prob = dist.log_prob(actions).sum()

        return actions.numpy(), log_prob

    def _compute_loss(self, trajectory):
        """REINFORCE with baseline (actor-critic) loss.

        Components:
        - Policy loss:  -log_prob * advantage
        - Value loss:   MSE(predicted_value, discounted_return)
        - Entropy bonus: encourages exploration
        """
        rewards = trajectory["rewards"]
        log_probs = torch.stack(trajectory["log_probs"])
        values = torch.stack(trajectory["values"])

        # Compute discounted returns
        returns = self._compute_returns(rewards)
        returns_tensor = torch.FloatTensor(returns)

        # Normalize returns for stable training
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        # Advantage = returns - baseline (value prediction)
        advantages = returns_tensor - values.detach()

        # Policy (actor) loss
        policy_loss = -(log_probs * advantages).mean()

        # Value (critic) loss
        value_loss = F.mse_loss(values, returns_tensor)

        # Entropy bonus for exploration
        obs_stack = torch.stack(trajectory["observations"])
        logits, _ = self.policy(obs_stack)
        num_slots = self.max_pending
        act_per_slot = self.max_nodes + 1
        logits_3d = logits.view(-1, num_slots, act_per_slot)
        probs = torch.softmax(logits_3d, dim=-1)
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy().mean()

        # Combined loss
        loss = (
            policy_loss
            + self.value_coeff * value_loss
            - self.entropy_coeff * entropy
        )
        return loss

    def _compute_returns(self, rewards):
        """Compute discounted cumulative returns."""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns
