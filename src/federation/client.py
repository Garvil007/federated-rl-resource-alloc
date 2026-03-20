import ray
import torch
import torch.nn.functional as F
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
        self.gae_lambda = train_config.get("gae_lambda", 0.95)
        self.clip_epsilon = train_config.get("clip_epsilon", 0.2)
        self.ppo_epochs = train_config.get("ppo_epochs", 4)
        self.entropy_coeff = train_config.get("entropy_coeff", 0.01)
        self.value_coeff = train_config.get("value_coeff", 0.5)
        self.max_nodes = self.env.max_nodes
        self.max_pending = self.env.max_pending

        self.policy = PolicyNetwork(
            obs_dim, act_dim, hidden_dim, max_nodes=self.max_nodes
        )
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=train_config.get("lr", 3e-4),
        )
        total_steps = train_config.get("num_rounds", 100) * train_config.get(
            "local_epochs", 5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-5
        )

        self.global_weights = None  # For FedProx
        self.control_variate = None  # For Scaffold (per-client c_i)
        self.global_control_variate = None  # For Scaffold (global c)

    def set_weights(self, global_weights, global_control=None):
        """Receive global model weights (and optionally control variates) from server."""
        self.policy.load_state_dict(global_weights)
        self.global_weights = {k: v.clone() for k, v in global_weights.items()}
        if global_control is not None:
            self.global_control_variate = {
                k: v.clone() for k, v in global_control.items()
            }

    def train_local(self, num_epochs: int):
        """Train locally for E epochs using PPO, return updated weights."""
        total_samples = 0
        total_reward = 0.0
        total_loss = 0.0
        total_sla = 0.0
        mu = self.train_config.get("mu", 0.01)
        strategy = self.train_config.get("strategy", "fedavg")

        # Snapshot weights before training (for FedNova / Scaffold delta)
        weights_before = {k: v.clone() for k, v in self.policy.state_dict().items()}

        for epoch in range(num_epochs):
            # Collect rollout
            trajectory = self._collect_rollout()

            episode_reward = sum(trajectory["rewards"])
            episode_samples = len(trajectory["rewards"])

            # PPO update: multiple SGD passes on the same trajectory
            epoch_loss = self._ppo_update(trajectory, strategy, mu)

            total_samples += episode_samples
            total_reward += episode_reward
            total_loss += epoch_loss
            total_sla += trajectory["final_info"].get("sla_rate", 0.0)

        # Compute weight delta for FedNova normalization
        weights_after = self.policy.state_dict()
        weight_delta = {k: weights_after[k] - weights_before[k] for k in weights_before}

        # Update Scaffold control variate
        new_control = None
        if strategy == "scaffold" and self.global_control_variate is not None:
            new_control = self._update_control_variate(
                weights_before, weights_after, num_epochs
            )

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
            weight_delta,
            new_control,
            num_epochs,  # local steps for FedNova
        )

    def _collect_rollout(self):
        """Run one episode, collecting full trajectory for PPO."""
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
            with torch.no_grad():
                logits, value = self.policy(obs_tensor)

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

        # Bootstrap value for GAE if episode didn't terminate
        if not (terminated or truncated):
            with torch.no_grad():
                _, last_val = self.policy(torch.FloatTensor(obs).unsqueeze(0))
            trajectory["bootstrap_value"] = last_val.squeeze().item()
        else:
            trajectory["bootstrap_value"] = 0.0

        if not trajectory["final_info"]:
            trajectory["final_info"] = info

        return trajectory

    def _ppo_update(self, trajectory, strategy, mu):
        """PPO clipped objective with multiple SGD passes."""
        obs = torch.stack(trajectory["observations"])
        actions = torch.stack(trajectory["actions"])
        old_log_probs = torch.stack(trajectory["log_probs"]).detach()
        old_values = torch.stack(trajectory["values"]).detach()

        # Compute GAE advantages and returns
        advantages, returns = self._compute_gae(
            trajectory["rewards"],
            old_values,
            trajectory["bootstrap_value"],
        )

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        batch_size = len(obs)

        for k in range(self.ppo_epochs):
            # Shuffle indices for mini-batch
            indices = torch.randperm(batch_size)
            mini_batch_size = max(batch_size // 4, 16)

            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, batch_size)
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # Forward pass
                logits, values = self.policy(mb_obs)

                # Compute new log probs
                num_slots = self.max_pending
                act_per_slot = self.max_nodes + 1
                logits_3d = logits.view(-1, num_slots, act_per_slot)
                probs = torch.softmax(logits_3d, dim=-1)
                dist = torch.distributions.Categorical(probs)

                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped too for stability)
                value_loss = F.mse_loss(values.squeeze(), mb_returns)

                # Combined loss
                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    - self.entropy_coeff * entropy
                )

                # FedProx proximal term
                if strategy == "fedprox" and self.global_weights is not None:
                    prox_term = sum(
                        torch.norm(p - self.global_weights[n]) ** 2
                        for n, p in self.policy.named_parameters()
                        if n in self.global_weights
                    )
                    loss += (mu / 2) * prox_term

                # Scaffold correction
                if (
                    strategy == "scaffold"
                    and self.control_variate is not None
                    and self.global_control_variate is not None
                ):
                    for n, p in self.policy.named_parameters():
                        if p.grad is not None and n in self.control_variate:
                            # Will be applied after backward
                            pass

                self.optimizer.zero_grad()
                loss.backward()

                # Scaffold: correct gradients with control variates
                if (
                    strategy == "scaffold"
                    and self.control_variate is not None
                    and self.global_control_variate is not None
                ):
                    for n, p in self.policy.named_parameters():
                        if p.grad is not None and n in self.control_variate:
                            p.grad += (
                                self.global_control_variate[n] - self.control_variate[n]
                            )

                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()

        self.scheduler.step()
        return total_loss / max(self.ppo_epochs, 1)

    def _compute_gae(self, rewards, values, bootstrap_value):
        """Generalized Advantage Estimation (GAE-λ).

        Blends TD(0) and Monte Carlo to reduce variance while keeping low bias.
        """
        rewards_t = torch.FloatTensor(rewards)
        T = len(rewards)
        advantages = torch.zeros(T)
        returns = torch.zeros(T)
        last_gae = 0.0

        for t in reversed(range(T)):
            next_value = bootstrap_value if t == T - 1 else values[t + 1]
            delta = rewards_t[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def _sample_action(self, logits):
        """Sample action from policy logits for MultiDiscrete action space."""
        num_slots = self.max_pending
        act_per_slot = self.max_nodes + 1
        logits_2d = logits.view(num_slots, act_per_slot)

        probs = torch.softmax(logits_2d, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()

        log_prob = dist.log_prob(actions).sum()
        return actions.numpy(), log_prob

    def _update_control_variate(self, weights_before, weights_after, num_steps):
        """Update per-client Scaffold control variate c_i.

        c_i_new = c_i - c + (w_global - w_local) / (η * K)
        """
        lr = self.optimizer.param_groups[0]["lr"]
        eta_K = lr * num_steps

        new_control = {}
        for key in weights_before:
            old_c = (
                self.control_variate[key]
                if self.control_variate
                else torch.zeros_like(weights_before[key])
            )
            global_c = self.global_control_variate.get(
                key, torch.zeros_like(weights_before[key])
            )
            new_control[key] = (
                old_c
                - global_c
                + (weights_before[key] - weights_after[key]) / max(eta_K, 1e-8)
            )

        self.control_variate = new_control
        return new_control
