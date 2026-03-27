# Federated RL Resource Allocation

> **Privacy-preserving federated reinforcement learning for edge computing resource allocation.**  
> Multiple heterogeneous edge clients collaboratively train a shared policy — without ever sharing raw data.

[![Python](https://img.shields.io/badge/Python-3.9%E2%80%933.11-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange?logo=pytorch)](https://pytorch.org/)
[![Ray](https://img.shields.io/badge/Ray-2.9%2B-teal?logo=ray)](https://www.ray.io/)
[![WandB](https://img.shields.io/badge/Weights%20%26%20Biases-Sweep-yellow?logo=weightsandbiases)](https://wandb.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Federated Strategies](#federated-strategies)
7. [Running a WandB Sweep](#running-a-wandb-sweep)
8. [Training Details](#training-details)
9. [Testing](#testing)
10. [Contributing](#contributing)
11. [License](#license)

---

## Overview

This project addresses the problem of **resource allocation in edge computing networks** — deciding which computational tasks should be assigned to which edge nodes to maximize throughput, minimize latency, satisfy SLA deadlines, and reduce energy consumption.

The solution uses **Federated Reinforcement Learning (FRL)**: instead of a single centralized agent with access to all data, multiple geographically distributed edge clients each train a local **PPO policy** on their own environment. A central federation server periodically aggregates these local models into a single improved global policy, without exposing any local data.

### Key Features

- 🌐 **Federated learning** — FedAvg, FedProx, FedNova, and Scaffold aggregation strategies
- 🤖 **PPO agent** with GAE (Generalized Advantage Estimation) and clipped surrogate objective
- ⚡ **Parallel client training** via [Ray](https://www.ray.io/) distributed computing
- 🧠 **Task-aware attention policy network** with residual connections for structured observation processing
- 📊 **Weights & Biases** sweep integration for automated hyperparameter optimization
- 🔧 **Hydra** configuration management for flexible experiment setup
- 📦 **Heterogeneous clients** — 5 clients simulating different deployment scenarios (data center, mobile edge, IoT, rural)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       Federation Server                          │
│   • Maintains the global PolicyNetwork (actor-critic)            │
│   • Aggregates client updates (FedAvg / FedProx / Scaffold)      │
│   • Logs all metrics to Weights & Biases                         │
└──────────────────┬──────────────────────────────────────────────┘
                   │  broadcast global weights
                   ▼
     ┌─────────────────────────────────┐
     │         5 FederatedClients      │  (parallel via Ray)
     │  Client 0: Small edge  (3 nodes)│
     │  Client 1: Data center(10 nodes)│
     │  Client 2: Mobile edge (4 nodes)│
     │  Client 3: IoT device  (6 nodes)│
     │  Client 4: Rural edge  (2 nodes)│
     └─────────────────────────────────┘
              │
              │  local PPO training on ResourceAllocationEnv
              ▼
     ┌─────────────────────────────────────────┐
     │         ResourceAllocationEnv           │
     │  • Gymnasium-compatible environment     │
     │  • Multi-discrete action space          │
     │    (assign each pending task → node)    │
     │  • Reward: throughput + SLA + energy    │
     └─────────────────────────────────────────┘
```

### Policy Network

The `PolicyNetwork` is a shared actor-critic with:
- **Shared backbone**: linear projection → LayerNorm → 3 Residual blocks
- **Task-aware attention**: cross-attention between node features and pending-task features
- **Actor head**: residual features + attention context → logits over `MultiDiscrete(11 × 10)` actions
- **Critic head**: decoupled linear pathway → scalar value estimate

---

## Project Structure

```
federated-rl-resource-alloc/
├── configs/
│   ├── training/
│   │   └── federated_5clients.yaml   # Main training configuration
│   └── sweep.yaml                    # WandB Bayesian sweep config
│
├── scripts/
│   └── train_federated.py            # Entry-point: Hydra + Ray + WandB
│
├── src/
│   ├── agents/
│   │   └── policy_network.py         # Actor-critic with residual + attention
│   ├── envs/
│   │   ├── __init__.py               # Registers ResourceAllocation-v0
│   │   └── resource_alloc_env.py     # Gymnasium edge computing environment
│   └── federation/
│       ├── client.py                 # PPO FederatedClient (Ray actor)
│       ├── server.py                 # FederationServer (Ray actor)
│       └── strategies.py            # FedAvg, FedProx, FedNova, Scaffold
│
├── tests/
│   ├── test_env.py                   # Environment compliance & smoke tests
│   └── test_federation.py            # Aggregation strategy unit tests
│
├── docker/                           # Docker & compose files
├── monitoring/                       # Prometheus & Grafana configs
├── pyproject.toml                    # Project metadata & dependencies
├── Makefile                          # Convenience commands
└── .pre-commit-config.yaml           # Ruff + mypy hooks
```

---

## Quick Start

### 1. Prerequisites

- Python 3.9–3.11
- Git

### 2. Clone & Install

```bash
git clone https://github.com/<your-username>/federated-rl-resource-alloc.git
cd federated-rl-resource-alloc

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install all dependencies
pip install -e ".[dev]"
```

### 3. Login to Weights & Biases

```bash
wandb login
```

### 4. Run Training

```bash
# Default config: 100 rounds, 5 clients, Scaffold strategy
python scripts/train_federated.py

# Override any config value with Hydra syntax
python scripts/train_federated.py federation.num_rounds=50 training.lr=0.001
```

---

## Configuration

The main config file is `configs/training/federated_5clients.yaml`:

```yaml
federation:
  num_clients: 5
  num_rounds: 100          # How many global aggregation rounds
  local_epochs: 5          # Local PPO epochs per round per client
  strategy: "scaffold"     # Aggregation strategy: fedavg | fedprox | fednova | scaffold
  mu: 0.01                 # Proximal term weight (FedProx only)

training:
  lr: 0.0003
  gamma: 0.99              # Discount factor
  gae_lambda: 0.95         # GAE-λ
  clip_epsilon: 0.2        # PPO clip ratio
  ppo_epochs: 4            # SGD passes per trajectory
  entropy_coeff: 0.05      # Exploration bonus weight
  rollouts_per_epoch: 3    # Episodes collected per local epoch
  obs_dim: 70              # Padded: 10 nodes × 4 + 10 tasks × 3
  act_dim: 110             # Padded: (10 nodes + 1) × 10 tasks
  hidden_dim: 256
```

Each `client_envs` entry configures a heterogeneous edge site:

| Client | Nodes | Traffic | Scenario          |
|--------|-------|---------|-------------------|
| 0      | 3     | Light   | Small edge        |
| 1      | 10    | Heavy   | Large data center |
| 2      | 4     | Bursty  | Mobile edge       |
| 3      | 6     | Steady  | Industrial IoT    |
| 4      | 2     | Sparse  | Rural edge        |

---

## Federated Strategies

| Strategy  | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| **FedAvg**   | Weighted average of client weights by number of samples                  |
| **FedProx**  | FedAvg + proximal term `μ/2 ‖w - w_global‖²` penalizing client drift    |
| **FedNova**  | Normalizes each client's update by local SGD steps for fairness          |
| **Scaffold** | Corrects client drift using per-client and global control variates       |

---

## Running a WandB Sweep

```bash
# 1. Create a new sweep
wandb sweep configs/sweep.yaml

# 2. Launch an agent with the returned sweep ID
wandb agent <entity>/<project>/<sweep-id>
```

The sweep uses **Bayesian optimization** to search over:
- `federation.num_rounds`: `[50, 100, 200]`
- `federation.local_epochs`: `[3, 5, 10]`
- `federation.strategy`: `fedavg | fedprox | scaffold`
- `federation.mu`: `[0.001, 0.1]`
- `training.lr`: log-uniform `[1e-5, 1e-2]`
- `training.entropy_coeff`: log-uniform `[0.005, 0.1]`

---

## Training Details

### Observation Space (`obs_dim = 70`)

Each observation is a concatenation of:
- **Node features** (padded to `max_nodes = 10`): `[cpu_util, mem_util, bw_util, queue_len]` × 10
- **Task features** (padded to `max_pending = 10`): `[cpu_demand, mem_demand, deadline]` × 10

### Action Space (`MultiDiscrete([11] × 10)`)

For each pending task slot, the agent outputs one of:
- `0 … 9`: assign task to node *i*
- `10`: reject / skip the task

### Reward Function

```python
r = w_throughput          # +0.5 per successful placement
  + w_sla × sla_ok        # +1.0 if latency ≤ deadline
  + w_latency × latency   # −0.15 × queue+processing delay
  + w_energy × energy     # −0.05 × CPU cost
  + w_util × |util−0.6|   # ±0.2 utilization balance bonus
  # OR
  + w_drop                # −0.5 per rejected/failed task
```

Rewards are clipped to `[-1.0, +1.0]` and normalized via a running mean/std before PPO updates.

---

## Testing

```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Tests cover:
- `test_env.py` — Gymnasium compliance, observation shapes, episode termination, seed determinism
- `test_federation.py` — FedAvg correctness (equal and weighted sample averaging), weight divergence metrics

---

## Contributing

1. Fork the repository and create a feature branch: `git checkout -b feature/my-improvement`
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Install pre-commit hooks: `pre-commit install`
4. Make your changes and ensure all checks pass: `pytest tests/ -v && ruff format . && ruff check .`
5. Open a pull request against the `develop` branch.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
