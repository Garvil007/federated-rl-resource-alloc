import ray
import torch
import time
from src.federation.strategies import (
    fed_avg,
    fed_prox,
    fed_nova,
    scaffold_aggregate,
    compute_weight_divergence,
)
from src.federation.client import FederatedClient


@ray.remote
class FederationServer:
    def __init__(self, config):
        self.config = config
        self.num_clients = config["federation"]["num_clients"]
        self.num_rounds = config["federation"]["num_rounds"]
        self.local_epochs = config["federation"]["local_epochs"]
        self.strategy = config["federation"].get("strategy", "scaffold")
        self.mu = config["federation"].get("mu", 0.01)
        self.global_model = self._init_global_model()
        self.history = []

        # Scaffold: initialize global control variate
        self.global_control = {
            k: torch.zeros_like(v) for k, v in self.global_model.state_dict().items()
        }

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

            # 1. Distribute global model (and control variates for Scaffold)
            global_weights = self.global_model.state_dict()
            gc = self.global_control if self.strategy == "scaffold" else None
            set_futures = [
                c.set_weights.remote(global_weights, gc) for c in self.clients
            ]
            ray.get(set_futures)

            # 2. Local training on each client (parallel via Ray)
            train_futures = [
                c.train_local.remote(self.local_epochs) for c in self.clients
            ]
            client_results = ray.get(train_futures)
            # client_results: [(weights, n_samples, metrics, delta, control, steps), ...]

            # 3. Aggregate based on strategy
            new_global = self._aggregate(global_weights, client_results)
            self.global_model.load_state_dict(new_global)

            # 4. Compute metrics
            round_duration = time.time() - round_start
            client_rewards = [m["reward"] for _, _, m, _, _, _ in client_results]
            avg_reward = sum(client_rewards) / len(client_rewards)

            client_slas = [m["sla_rate"] for _, _, m, _, _, _ in client_results]
            avg_sla = sum(client_slas) / len(client_slas)

            divergences = [
                compute_weight_divergence(global_weights, w)
                for w, _, _, _, _, _ in client_results
            ]
            avg_divergence = sum(divergences) / len(divergences)

            round_metrics = {
                "round": round_num,
                "global/avg_reward": avg_reward,
                "global/reward_std": float(torch.std(torch.tensor(client_rewards))),
                "global/avg_sla_rate": avg_sla,
                "global/weight_divergence": avg_divergence,
                "global/round_duration_s": round_duration,
                "global/strategy": self.strategy,
            }

            for cid, (_, _, metrics, _, _, _) in enumerate(client_results):
                round_metrics[f"client_{cid}/reward"] = metrics["reward"]
                round_metrics[f"client_{cid}/sla_rate"] = metrics["sla_rate"]
                round_metrics[f"client_{cid}/loss"] = metrics["loss"]

            self.history.append(round_metrics)

            print(
                f"Round {round_num}/{self.num_rounds} | "
                f"Strategy: {self.strategy} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"SLA: {avg_sla:.2%} | "
                f"Divergence: {avg_divergence:.4f} | "
                f"Duration: {round_duration:.1f}s"
            )

        return self.global_model.state_dict(), self.history

    def _aggregate(self, global_weights, client_results):
        """Dispatch to the correct aggregation strategy."""
        weight_pairs = [(w, n) for w, n, _, _, _, _ in client_results]

        if self.strategy == "fedavg":
            return fed_avg(global_weights, weight_pairs)

        elif self.strategy == "fedprox":
            return fed_prox(global_weights, weight_pairs, self.mu)

        elif self.strategy == "fednova":
            delta_tuples = [
                (delta, n, steps) for _, n, _, delta, _, steps in client_results
            ]
            return fed_nova(global_weights, delta_tuples)

        elif self.strategy == "scaffold":
            client_controls = [ctrl for _, _, _, _, ctrl, _ in client_results]
            new_weights, new_gc = scaffold_aggregate(
                global_weights,
                weight_pairs,
                client_controls,
                self.global_control,
                self.num_clients,
            )
            self.global_control = new_gc
            return new_weights

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
