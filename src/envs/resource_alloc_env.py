import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class EdgeNode:
    """Represents a single edge computing node."""

    cpu_capacity: float = 100.0  # Total CPU units
    mem_capacity: float = 64.0  # GB RAM
    bw_capacity: float = 1000.0  # Mbps
    cpu_used: float = 0.0
    mem_used: float = 0.0
    bw_used: float = 0.0
    task_queue: list = field(default_factory=list)
    energy_per_cpu: float = 0.5  # Watts per CPU unit

    @property
    def cpu_util(self) -> float:
        return self.cpu_used / self.cpu_capacity

    @property
    def mem_util(self) -> float:
        return self.mem_used / self.mem_capacity

    @property
    def can_accept(self) -> bool:
        return self.cpu_util < 0.95 and self.mem_util < 0.95


class ResourceAllocationEnv(gym.Env):
    """
    Edge Computing Resource Allocation Environment.

    The agent allocates incoming tasks to edge nodes,
    balancing throughput, latency, SLA compliance,
    and energy efficiency.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        cfg = config or {}
        self.num_nodes = cfg.get("num_nodes", 5)
        self.max_pending = cfg.get("max_pending_tasks", 10)
        self.episode_length = cfg.get("episode_length", 1000)
        self.task_arrival_rate = cfg.get("task_arrival_rate", 3.0)
        self.seed_val = cfg.get("seed", 42)

        self.max_nodes = cfg.get("max_nodes", 10)
        
        # Observation: [node_features(max_nodes*4) + task_features(max_pending*3)]
        obs_dim = self.max_nodes * 4 + self.max_pending * 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: assign each pending task to a node (or reject)
        self.action_space = spaces.MultiDiscrete(
            [self.max_nodes + 1] * self.max_pending
        )

        # Reward weights (tunable hyperparameters)
        self.w_throughput = cfg.get("w_throughput", 1.0)
        self.w_latency = cfg.get("w_latency", -0.3)
        self.w_energy = cfg.get("w_energy", -0.1)
        self.w_sla = cfg.get("w_sla", 2.0)
        self.w_drop = cfg.get("w_drop", -2.0)
        self.w_util = cfg.get("w_util", 0.3)  # utilization bonus

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed if seed else self.seed_val)
        self.nodes = [
            EdgeNode(
                cpu_capacity=self.np_random.uniform(50, 150),
                mem_capacity=self.np_random.uniform(32, 128),
                bw_capacity=self.np_random.uniform(500, 2000),
            )
            for _ in range(self.num_nodes)
        ]
        self.current_step = 0
        self.pending_tasks = self._generate_tasks()
        self.total_completed = 0
        self.total_dropped = 0
        self.total_sla_met = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        rewards = []
        for task_idx, node_idx in enumerate(action):
            if task_idx >= len(self.pending_tasks):
                break
            task = self.pending_tasks[task_idx]

            if node_idx >= self.num_nodes:  # reject or invalid padded node
                self.total_dropped += 1
                rewards.append(self.w_drop)
            elif self.nodes[node_idx].can_accept:
                self._assign_task(node_idx, task)
                latency = self._estimate_latency(node_idx)
                sla_ok = latency <= task["deadline"]
                self.total_completed += 1
                if sla_ok:
                    self.total_sla_met += 1
                energy = self.nodes[node_idx].energy_per_cpu * task["cpu"]
                # Utilization bonus: reward balanced usage (penalize extreme under/over)
                util = self.nodes[node_idx].cpu_util
                util_bonus = self.w_util * (1.0 - abs(util - 0.6))  # sweet spot ~60%
                r = (
                    self.w_throughput
                    + self.w_latency * latency
                    + self.w_energy * energy
                    + self.w_sla * float(sla_ok)
                    + util_bonus
                )
                rewards.append(r)
            else:
                self.total_dropped += 1
                rewards.append(self.w_drop)

        self._process_queues()  # Advance time, free resources
        self.current_step += 1
        self.pending_tasks = self._generate_tasks()

        terminated = self.current_step >= self.episode_length
        return (
            self._get_obs(),
            sum(rewards),
            terminated,
            False,  # truncated
            self._get_info(),
        )

    def _get_obs(self) -> np.ndarray:
        node_features = []
        for node in self.nodes:
            node_features.extend(
                [
                    np.clip(node.cpu_util, 0.0, 1.0),
                    np.clip(node.mem_util, 0.0, 1.0),
                    np.clip(node.bw_used / node.bw_capacity, 0.0, 1.0),
                    np.clip(len(node.task_queue) / 20.0, 0.0, 1.0),
                ]
            )

        pad_nodes = self.max_nodes - len(self.nodes)
        if pad_nodes > 0:
            node_features.extend([0.0] * (pad_nodes * 4))

        # Compute max capacities from actual nodes for normalization
        max_cpu = max((n.cpu_capacity for n in self.nodes), default=100.0)
        max_mem = max((n.mem_capacity for n in self.nodes), default=64.0)

        task_features = []
        for i in range(self.max_pending):
            if i < len(self.pending_tasks):
                t = self.pending_tasks[i]
                task_features.extend(
                    [
                        np.clip(t["cpu"] / max_cpu, 0.0, 1.0),
                        np.clip(t["mem"] / max_mem, 0.0, 1.0),
                        np.clip(t["deadline"] / 10.0, 0.0, 1.0),
                    ]
                )
            else:
                task_features.extend([0.0, 0.0, 0.0])

        return np.array(node_features + task_features, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self.current_step,
            "completed": self.total_completed,
            "dropped": self.total_dropped,
            "sla_met": self.total_sla_met,
            "sla_rate": (self.total_sla_met / max(self.total_completed, 1)),
            "utilization": np.mean([n.cpu_util for n in self.nodes]),
        }

    def _generate_tasks(self) -> list:
        n_tasks = self.np_random.poisson(self.task_arrival_rate)
        n_tasks = min(n_tasks, self.max_pending)
        tasks = []
        for _ in range(n_tasks):
            tasks.append(
                {
                    "cpu": float(self.np_random.uniform(5, 50)),
                    "mem": float(self.np_random.uniform(1, 32)),
                    "deadline": float(self.np_random.uniform(1, 10)),
                    "arrival_time": self.current_step,
                }
            )
        return tasks

    def _assign_task(self, node_idx: int, task: dict):
        node = self.nodes[node_idx]
        node.cpu_used += task["cpu"]
        node.mem_used += task["mem"]
        node.task_queue.append(
            {
                **task,
                "remaining_time": self.np_random.uniform(1, 5),
            }
        )

    def _estimate_latency(self, node_idx: int) -> float:
        node = self.nodes[node_idx]
        queue_delay = len(node.task_queue) * 0.1
        processing_delay = node.cpu_util * 2.0
        return queue_delay + processing_delay

    def _process_queues(self):
        for node in self.nodes:
            completed = []
            for i, task in enumerate(node.task_queue):
                task["remaining_time"] -= 1
                if task["remaining_time"] <= 0:
                    completed.append(i)
                    node.cpu_used = max(0, node.cpu_used - task["cpu"])
                    node.mem_used = max(0, node.mem_used - task["mem"])
            for i in reversed(completed):
                node.task_queue.pop(i)

    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.current_step}/{self.episode_length}")
            for i, node in enumerate(self.nodes):
                print(
                    f"  Node {i}: CPU={node.cpu_util:.1%} MEM={node.mem_util:.1%} Queue={len(node.task_queue)}"
                )
