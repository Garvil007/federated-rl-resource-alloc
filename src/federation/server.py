import ray
import torch
import wandb
import time
from src.federation.strategies import fed_avg, fed_prox, compute_weight_divergence
from src.federation.client import FederatedClient


@ray.remote
class FederationServer:
    def __init__(self, config):
        self.config = config
        self.num_clients = config["federation"]["num_clients"]
        self.num_rounds = config["federation"]["num_rounds"]
        self.local_epochs = config["federation"]["local_epochs"]
        self.strategy = config["federation"].get("strategy", "fedavg")
        self.mu = config["federation"].get("mu", 0.01)
        self.global_model = self._init_global_model()

        # Create heterogeneous edge clients
        self.clients = [
            FederatedClient.remote(
                client_id=i,
                env_config=config["client_envs"][i],
                train_config=config["training"],
            )
            for i in range(self.num_clients)
        ]

    def _init_global_model(self):
        """Initialize the global policy network."""
        from src.agents.policy_network import PolicyNetwork

        obs_dim = self.config["training"]["obs_dim"]
        act_dim = self.config["training"]["act_dim"]
        return PolicyNetwork(obs_dim, act_dim)

    def run_federation(self):
        """Main federated learning loop."""
        for round_num in range(self.num_rounds):
            round_start = time.time()

            # 1. Distribute global model to all clients
            global_weights = self.global_model.state_dict()
            set_futures = [c.set_weights.remote(global_weights) for c in self.clients]
            ray.get(set_futures)

            # 2. Local training on each client (parallel via Ray)
            train_futures = [
                c.train_local.remote(self.local_epochs) for c in self.clients
            ]
            client_results = ray.get(train_futures)
            # client_results: [(weights, n_samples, metrics), ...]

            # 3. Aggregate
            weight_pairs = [(w, n) for w, n, _ in client_results]
            if self.strategy == "fedavg":
                new_global = fed_avg(global_weights, weight_pairs)
            elif self.strategy == "fedprox":
                new_global = fed_prox(global_weights, weight_pairs, self.mu)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            self.global_model.load_state_dict(new_global)

            # 4. Compute metrics
            round_duration = time.time() - round_start
            client_rewards = [m["reward"] for _, _, m in client_results]
            avg_reward = sum(client_rewards) / len(client_rewards)

            # Weight divergence
            divergences = [
                compute_weight_divergence(global_weights, w)
                for w, _, _ in client_results
            ]
            avg_divergence = sum(divergences) / len(divergences)

            # 5. Log aggregated metrics
            wandb.log(
                {
                    "round": round_num,
                    "global/avg_reward": avg_reward,
                    "global/reward_std": float(torch.std(torch.tensor(client_rewards))),
                    "global/weight_divergence": avg_divergence,
                    "global/round_duration_s": round_duration,
                }
            )

            # Log per-client metrics
            for cid, (_, _, metrics) in enumerate(client_results):
                wandb.log(
                    {
                        "round": round_num,
                        f"client_{cid}/reward": metrics["reward"],
                        f"client_{cid}/sla_rate": metrics["sla_rate"],
                        f"client_{cid}/loss": metrics["loss"],
                    }
                )

            print(
                f"Round {round_num}/{self.num_rounds} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Divergence: {avg_divergence:.4f} | "
                f"Duration: {round_duration:.1f}s"
            )

        return self.global_model.state_dict()
