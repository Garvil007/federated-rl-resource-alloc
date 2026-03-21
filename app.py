"""
Federated RL Resource Allocation — Visualization Dashboard
===========================================================
A comprehensive Streamlit app that visualizes every component
of the Federated RL pipeline in real-time.

Run: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
import random
import math

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Federated RL Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CUSTOM CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

    .main { background-color: #0a0e17; }
    .stApp { background-color: #0a0e17; }

    h1, h2, h3 { font-family: 'Plus Jakarta Sans', sans-serif !important; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1220 100%);
        border: 1px solid #2a3050;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #8892a8;
        font-size: 0.85rem;
        margin-top: 4px;
    }

    .phase-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.5px;
    }

    .status-running { background: #0f2d1f; color: #34d399; border: 1px solid #065f46; }
    .status-complete { background: #1a1a3e; color: #818cf8; border: 1px solid #3730a3; }
    .status-waiting { background: #2a1f0d; color: #fbbf24; border: 1px solid #92400e; }

    .code-block {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #c9d1d9;
        overflow-x: auto;
    }

    .arch-layer {
        border-radius: 10px;
        padding: 16px;
        margin: 6px 0;
        border-left: 4px solid;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1220 0%, #0a0e17 100%);
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIMULATION CLASSES (mirrors the real project)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EdgeNode:
    node_id: int
    cpu_capacity: float
    mem_capacity: float
    bw_capacity: float
    cpu_used: float = 0.0
    mem_used: float = 0.0
    bw_used: float = 0.0
    queue_len: int = 0
    energy_per_cpu: float = 0.5

    @property
    def cpu_util(self): return self.cpu_used / self.cpu_capacity
    @property
    def mem_util(self): return self.mem_used / self.mem_capacity
    @property
    def bw_util(self): return self.bw_used / self.bw_capacity


@dataclass
class Task:
    cpu_req: float
    mem_req: float
    deadline: float
    arrival_time: int


@dataclass
class ClientConfig:
    client_id: int
    name: str
    num_nodes: int
    task_arrival_rate: float
    cpu_range: Tuple[float, float]
    mem_range: Tuple[float, float]
    color: str


# Default client configurations (heterogeneous)
DEFAULT_CLIENTS = [
    ClientConfig(0, "Small Edge Site",     3,  1.5, (30, 80),   (16, 32),  "#34d399"),
    ClientConfig(1, "Large Data Center",   10, 8.0, (100, 200), (64, 256), "#818cf8"),
    ClientConfig(2, "Mobile Edge",         4,  5.0, (20, 60),   (8, 16),   "#fbbf24"),
    ClientConfig(3, "Industrial IoT",      6,  3.0, (50, 100),  (32, 64),  "#f87171"),
    ClientConfig(4, "Rural Edge",          2,  0.8, (20, 50),   (8, 16),   "#a78bfa"),
]


def simulate_environment_step(nodes: List[EdgeNode], rng: np.random.Generator):
    """Simulate one timestep of the resource allocation environment."""
    for node in nodes:
        # Fluctuate utilization
        node.cpu_used = np.clip(
            node.cpu_used + rng.normal(0, node.cpu_capacity * 0.05),
            0, node.cpu_capacity * 0.95
        )
        node.mem_used = np.clip(
            node.mem_used + rng.normal(0, node.mem_capacity * 0.03),
            0, node.mem_capacity * 0.95
        )
        node.bw_used = np.clip(
            node.bw_used + rng.normal(0, node.bw_capacity * 0.04),
            0, node.bw_capacity * 0.95
        )
        node.queue_len = max(0, node.queue_len + rng.integers(-2, 4))
    return nodes


def simulate_federation_round(
    round_num: int,
    num_clients: int,
    strategy: str,
    rng: np.random.Generator,
    prev_reward: float = 0.0,
) -> Dict:
    """Simulate metrics from one federation round."""
    # Rewards improve over time with noise
    improvement = 0.5 * math.log(round_num + 1)
    base_reward = -15 + improvement + rng.normal(0, 2)
    global_reward = max(prev_reward * 0.7 + base_reward * 0.3, prev_reward + rng.normal(0.3, 0.5))

    client_rewards = []
    client_losses = []
    client_sla_rates = []
    for c in range(num_clients):
        cr = global_reward + rng.normal(0, 3)  # Client-specific noise
        client_rewards.append(cr)
        client_losses.append(max(0.01, 2.0 - 0.015 * round_num + rng.normal(0, 0.2)))
        client_sla_rates.append(np.clip(0.4 + 0.005 * round_num + rng.normal(0, 0.05), 0, 1))

    divergence = max(0, 5.0 - 0.03 * round_num + rng.normal(0, 0.5))
    if strategy == "FedProx":
        divergence *= 0.6  # FedProx reduces divergence

    return {
        "round": round_num,
        "global_reward": global_reward,
        "client_rewards": client_rewards,
        "client_losses": client_losses,
        "client_sla_rates": client_sla_rates,
        "weight_divergence": divergence,
        "round_duration": 5.0 + rng.exponential(3.0),
        "communication_cost_mb": num_clients * (2.5 + rng.exponential(0.5)),
    }


def simulate_serving_metrics(rng: np.random.Generator, n_points: int = 100) -> pd.DataFrame:
    """Simulate model serving metrics."""
    timestamps = pd.date_range("2026-01-01", periods=n_points, freq="1min")
    latencies = np.abs(rng.normal(0.015, 0.008, n_points))
    throughput = 100 + rng.normal(0, 15, n_points).cumsum() * 0.1 + np.arange(n_points) * 0.5
    errors = rng.poisson(0.5, n_points)
    return pd.DataFrame({
        "timestamp": timestamps,
        "latency_ms": latencies * 1000,
        "throughput_rps": np.clip(throughput, 10, 500),
        "errors": errors,
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "rng" not in st.session_state:
    st.session_state.rng = np.random.default_rng(42)
if "federation_history" not in st.session_state:
    st.session_state.federation_history = []
if "current_round" not in st.session_state:
    st.session_state.current_round = 0
if "is_running" not in st.session_state:
    st.session_state.is_running = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("## 🤖 Fed-RL Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Project Overview",
            "🌍 Environment Visualizer",
            "🔄 Federation Simulator",
            "📊 Training Analytics",
            "🚀 Model Serving Monitor",
            "⚙️ CI/CD Pipeline View",
            "🏗️ Architecture Explorer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Simulation Config")

    num_clients = st.slider("Number of Clients", 2, 10, 5)
    num_rounds = st.slider("Federation Rounds", 10, 200, 50)
    strategy = st.selectbox("Aggregation Strategy", ["FedAvg", "FedProx"])
    local_epochs = st.slider("Local Epochs", 1, 20, 5)
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01],
        value=0.0003,
        format_func=lambda x: f"{x:.5f}",
    )

    if strategy == "FedProx":
        mu = st.slider("FedProx μ (proximal term)", 0.001, 0.1, 0.01, step=0.001)
    else:
        mu = 0.0

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#555; font-size:0.75rem;'>"
        "Built for Federated RL<br/>Resource Allocation Project"
        "</div>",
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: PROJECT OVERVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "🏠 Project Overview":
    st.markdown("# 🤖 Federated RL for Edge Resource Allocation")
    st.markdown("#### A complete visualization of what every component does")
    st.markdown("---")

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">6</div>
            <div class="metric-label">Architecture Layers</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">5+</div>
            <div class="metric-label">Edge Clients</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">CI/CD Pipelines</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">9</div>
            <div class="metric-label">Major Components</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 How The Project Works (End-to-End)")

    steps = [
        ("1️⃣", "Custom Gym Environment", "Simulates edge computing with N nodes, task arrivals, SLA deadlines", "status-complete"),
        ("2️⃣", "PPO Agent (Ray/RLlib)", "Neural network learns to allocate tasks to nodes optimally", "status-complete"),
        ("3️⃣", "Federated Training Loop", "K clients train locally, server aggregates via FedAvg/FedProx", "status-running"),
        ("4️⃣", "W&B Experiment Tracking", "Live dashboards, per-client metrics, hyperparameter sweeps", "status-running"),
        ("5️⃣", "MLflow Model Registry", "Version models, promote Staging → Production", "status-waiting"),
        ("6️⃣", "Prometheus + Grafana", "Real-time monitoring, latency histograms, alert rules", "status-waiting"),
        ("7️⃣", "GitHub Actions CI/CD", "Auto lint/test/train/deploy on every push", "status-waiting"),
        ("8️⃣", "Docker Deployment", "Containerized serving with health checks", "status-waiting"),
    ]

    for emoji, title, desc, status in steps:
        col1, col2 = st.columns([1, 12])
        with col1:
            st.markdown(f"### {emoji}")
        with col2:
            st.markdown(f"**{title}** — {desc}")

    st.markdown("---")
    st.markdown("### 🧭 Use the sidebar to explore each component visually →")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: ENVIRONMENT VISUALIZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🌍 Environment Visualizer":
    st.markdown("# 🌍 Resource Allocation Environment")
    st.markdown("*Visualize what the RL agent 'sees' and 'does' at each timestep*")
    st.markdown("---")

    # Select client to visualize
    client_idx = st.selectbox(
        "Select Edge Client to Visualize",
        range(min(num_clients, len(DEFAULT_CLIENTS))),
        format_func=lambda i: f"Client {i}: {DEFAULT_CLIENTS[i].name} ({DEFAULT_CLIENTS[i].num_nodes} nodes, λ={DEFAULT_CLIENTS[i].task_arrival_rate})"
    )

    client = DEFAULT_CLIENTS[client_idx]
    rng = st.session_state.rng

    # Generate nodes for this client
    nodes = [
        EdgeNode(
            node_id=i,
            cpu_capacity=rng.uniform(*client.cpu_range),
            mem_capacity=rng.uniform(*client.mem_range),
            bw_capacity=rng.uniform(500, 2000),
            cpu_used=rng.uniform(0, 0.7) * rng.uniform(*client.cpu_range),
            mem_used=rng.uniform(0, 0.6) * rng.uniform(*client.mem_range),
            bw_used=rng.uniform(100, 800),
            queue_len=rng.integers(0, 15),
        )
        for i in range(client.num_nodes)
    ]

    # Generate pending tasks
    n_tasks = rng.poisson(client.task_arrival_rate)
    tasks = [
        Task(
            cpu_req=float(rng.uniform(5, 50)),
            mem_req=float(rng.uniform(1, 32)),
            deadline=float(rng.uniform(1, 10)),
            arrival_time=0,
        )
        for _ in range(n_tasks)
    ]

    # ── Node Status ──
    st.markdown("### 🖥️ Edge Nodes — Current State")

    node_df = pd.DataFrame([
        {
            "Node": f"Node {n.node_id}",
            "CPU Util (%)": round(n.cpu_util * 100, 1),
            "Mem Util (%)": round(n.mem_util * 100, 1),
            "BW Util (%)": round(n.bw_util * 100, 1),
            "Queue": n.queue_len,
            "CPU Cap": f"{n.cpu_capacity:.0f}",
            "Mem Cap": f"{n.mem_capacity:.0f} GB",
        }
        for n in nodes
    ])

    # Heatmap of utilization
    fig_heat = go.Figure()
    metrics_names = ["CPU", "Memory", "Bandwidth"]
    z_data = [
        [n.cpu_util * 100 for n in nodes],
        [n.mem_util * 100 for n in nodes],
        [n.bw_util * 100 for n in nodes],
    ]
    fig_heat = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[f"Node {n.node_id}" for n in nodes],
        y=metrics_names,
        colorscale=[[0, "#0d1117"], [0.5, "#2196F3"], [0.8, "#ff9800"], [1, "#f44336"]],
        zmin=0, zmax=100,
        text=[[f"{v:.0f}%" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        hovertemplate="Node: %{x}<br>Metric: %{y}<br>Utilization: %{z:.1f}%<extra></extra>",
    ))
    fig_heat.update_layout(
        title="Node Utilization Heatmap",
        height=250,
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#0d1117",
        margin=dict(l=80, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Pending Tasks ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Pending Tasks")
        if tasks:
            task_df = pd.DataFrame([
                {
                    "Task": f"Task {i}",
                    "CPU Required": f"{t.cpu_req:.1f}",
                    "Memory Required": f"{t.mem_req:.1f} GB",
                    "Deadline": f"{t.deadline:.1f}s",
                }
                for i, t in enumerate(tasks)
            ])
            st.dataframe(task_df, use_container_width=True, hide_index=True)
        else:
            st.info("No pending tasks this timestep")

    with col2:
        st.markdown("### 🎯 Agent's Action (Simulated)")
        if tasks:
            actions = [rng.integers(0, client.num_nodes + 1) for _ in tasks]
            action_df = pd.DataFrame([
                {
                    "Task": f"Task {i}",
                    "Decision": f"→ Node {a}" if a < client.num_nodes else "❌ REJECT",
                    "Reason": "Capacity available" if a < client.num_nodes else "All nodes overloaded",
                }
                for i, a in enumerate(actions)
            ])
            st.dataframe(action_df, use_container_width=True, hide_index=True)

    # ── Reward Breakdown ──
    st.markdown("### 💰 Reward Computation (This Timestep)")
    fig_rew = go.Figure()
    components = ["Throughput (+1.0)", "SLA Bonus (+2.0)", "Latency Penalty", "Energy Penalty", "Drop Penalty"]
    values = [
        len([a for a in actions if a < client.num_nodes]) * 1.0,
        len([a for a in actions if a < client.num_nodes]) * 2.0 * 0.7,
        -sum([rng.uniform(0.1, 0.5) for a in actions if a < client.num_nodes]) * 0.5,
        -sum([rng.uniform(0.05, 0.2) for a in actions if a < client.num_nodes]) * 0.2,
        -len([a for a in actions if a >= client.num_nodes]) * 3.0,
    ]
    colors = ["#34d399", "#818cf8", "#f87171", "#fbbf24", "#ef4444"]
    fig_rew = go.Figure(go.Bar(
        x=values, y=components, orientation='h',
        marker_color=colors,
        text=[f"{v:+.2f}" for v in values],
        textposition="outside",
    ))
    fig_rew.update_layout(
        title=f"Total Reward: {sum(values):+.2f}",
        height=280,
        template="plotly_dark",
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#0d1117",
        margin=dict(l=160, r=60, t=40, b=20),
        xaxis_title="Reward Component Value",
    )
    st.plotly_chart(fig_rew, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: FEDERATION SIMULATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🔄 Federation Simulator":
    st.markdown("# 🔄 Federated Training Simulator")
    st.markdown(f"*{strategy} with {num_clients} clients × {num_rounds} rounds × {local_epochs} local epochs*")
    st.markdown("---")

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        run_sim = st.button("▶️ Run Full Simulation", type="primary", use_container_width=True)
    with col_btn2:
        step_sim = st.button("⏭️ Step One Round", use_container_width=True)
    with col_btn3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.federation_history = []
            st.session_state.current_round = 0
            st.rerun()

    rng = st.session_state.rng

    if run_sim:
        st.session_state.federation_history = []
        progress = st.progress(0, text="Starting federation...")
        prev_reward = -15.0

        for r in range(num_rounds):
            result = simulate_federation_round(r, num_clients, strategy, rng, prev_reward)
            st.session_state.federation_history.append(result)
            prev_reward = result["global_reward"]
            progress.progress((r + 1) / num_rounds, text=f"Round {r+1}/{num_rounds} | Reward: {prev_reward:.2f}")

        st.session_state.current_round = num_rounds
        progress.empty()
        st.success(f"✅ Completed {num_rounds} rounds!")

    if step_sim:
        r = st.session_state.current_round
        prev = st.session_state.federation_history[-1]["global_reward"] if st.session_state.federation_history else -15.0
        result = simulate_federation_round(r, num_clients, strategy, rng, prev)
        st.session_state.federation_history.append(result)
        st.session_state.current_round += 1

    # ── Visualize Results ──
    history = st.session_state.federation_history

    if history:
        st.markdown("---")

        # Top-level metrics
        latest = history[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Round", f"{latest['round']}/{num_rounds}", delta=f"+1")
        m2.metric("Global Reward", f"{latest['global_reward']:.2f}",
                  delta=f"{latest['global_reward'] - history[-2]['global_reward']:.2f}" if len(history) > 1 else None)
        m3.metric("Weight Divergence", f"{latest['weight_divergence']:.3f}",
                  delta=f"{latest['weight_divergence'] - history[-2]['weight_divergence']:.3f}" if len(history) > 1 else None,
                  delta_color="inverse")
        m4.metric("Comm. Cost", f"{latest['communication_cost_mb']:.1f} MB")

        # Global reward over rounds
        fig_reward = go.Figure()
        rounds = [h["round"] for h in history]
        rewards = [h["global_reward"] for h in history]

        fig_reward.add_trace(go.Scatter(
            x=rounds, y=rewards,
            mode="lines+markers",
            name="Global Avg Reward",
            line=dict(color="#818cf8", width=3),
            marker=dict(size=4),
        ))

        # Per-client rewards
        for c in range(num_clients):
            c_rewards = [h["client_rewards"][c] if c < len(h["client_rewards"]) else 0 for h in history]
            fig_reward.add_trace(go.Scatter(
                x=rounds, y=c_rewards,
                mode="lines",
                name=f"Client {c}",
                line=dict(width=1, dash="dot"),
                opacity=0.5,
            ))

        fig_reward.update_layout(
            title="Reward Convergence — Global vs Per-Client",
            xaxis_title="Federation Round",
            yaxis_title="Episode Reward",
            height=400,
            template="plotly_dark",
            paper_bgcolor="#0a0e17",
            plot_bgcolor="#0d1117",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_reward, use_container_width=True)

        # Second row of charts
        col1, col2 = st.columns(2)

        with col1:
            # Weight divergence
            fig_div = go.Figure()
            divs = [h["weight_divergence"] for h in history]
            fig_div.add_trace(go.Scatter(
                x=rounds, y=divs,
                fill="tozeroy",
                fillcolor="rgba(249, 115, 22, 0.15)",
                line=dict(color="#f97316", width=2),
                name="Weight Divergence",
            ))
            fig_div.add_hline(y=5.0, line_dash="dash", line_color="#ef4444",
                             annotation_text="Alert Threshold")
            fig_div.update_layout(
                title="Client Weight Divergence",
                xaxis_title="Round", yaxis_title="L2 Distance",
                height=350, template="plotly_dark",
                paper_bgcolor="#0a0e17", plot_bgcolor="#0d1117",
            )
            st.plotly_chart(fig_div, use_container_width=True)

        with col2:
            # SLA rates per client
            fig_sla = go.Figure()
            for c in range(num_clients):
                sla_rates = [h["client_sla_rates"][c] if c < len(h["client_sla_rates"]) else 0 for h in history]
                fig_sla.add_trace(go.Scatter(
                    x=rounds, y=sla_rates,
                    mode="lines",
                    name=f"Client {c}: {DEFAULT_CLIENTS[c].name}" if c < len(DEFAULT_CLIENTS) else f"Client {c}",
                    line=dict(width=2),
                ))
            fig_sla.add_hline(y=0.7, line_dash="dash", line_color="#34d399",
                             annotation_text="Target SLA 70%")
            fig_sla.update_layout(
                title="SLA Compliance Rate by Client",
                xaxis_title="Round", yaxis_title="SLA Rate",
                height=350, template="plotly_dark",
                paper_bgcolor="#0a0e17", plot_bgcolor="#0d1117",
                legend=dict(font=dict(size=9)),
            )
            st.plotly_chart(fig_sla, use_container_width=True)

        # Third row: Loss + Round Duration
        col3, col4 = st.columns(2)

        with col3:
            fig_loss = go.Figure()
            for c in range(num_clients):
                losses = [h["client_losses"][c] if c < len(h["client_losses"]) else 0 for h in history]
                fig_loss.add_trace(go.Scatter(
                    x=rounds, y=losses, mode="lines", name=f"Client {c}",
                    line=dict(width=1.5),
                ))
            fig_loss.update_layout(
                title="Training Loss by Client",
                xaxis_title="Round", yaxis_title="Policy Loss",
                height=300, template="plotly_dark",
                paper_bgcolor="#0a0e17", plot_bgcolor="#0d1117",
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with col4:
            durations = [h["round_duration"] for h in history]
            fig_dur = go.Figure(go.Histogram(
                x=durations,
                nbinsx=20,
                marker_color="#818cf8",
            ))
            fig_dur.update_layout(
                title="Round Duration Distribution",
                xaxis_title="Duration (seconds)", yaxis_title="Count",
                height=300, template="plotly_dark",
                paper_bgcolor="#0a0e17", plot_bgcolor="#0d1117",
            )
            st.plotly_chart(fig_dur, use_container_width=True)

    else:
        st.info("Click **Run Full Simulation** or **Step One Round** to start!")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: TRAINING ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📊 Training Analytics":
    st.markdown("# 📊 Training Analytics — W&B Style")
    st.markdown("*What you'd see in your Weights & Biases dashboard*")
    st.markdown("---")

    # Generate comparison data: FedAvg vs FedProx vs Centralized
    rng = np.random.default_rng(42)
    rounds_range = np.arange(100)

    def gen_curve(base, improvement_rate, noise, rng):
        rewards = []
        r = base
        for i in rounds_range:
            r += improvement_rate * (1 / (i + 1)) + rng.normal(0, noise)
            rewards.append(r)
        return rewards

    fedavg_rewards = gen_curve(-15, 5.0, 1.5, rng)
    fedprox_rewards = gen_curve(-15, 5.5, 1.0, rng)
    central_rewards = gen_curve(-15, 6.0, 0.8, rng)

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=list(rounds_range), y=fedavg_rewards, name="FedAvg (5 clients)",
                                  line=dict(color="#818cf8", width=3)))
    fig_comp.add_trace(go.Scatter(x=list(rounds_range), y=fedprox_rewards, name="FedProx (5 clients, μ=0.01)",
                                  line=dict(color="#34d399", width=3)))
    fig_comp.add_trace(go.Scatter(x=list(rounds_range), y=central_rewards, name="Centralized Baseline",
                                  line=dict(color="#fbbf24", width=3, dash="dash")))
    fig_comp.update_layout(
        title="Strategy Comparison: FedAvg vs FedProx vs Centralized",
        xaxis_title="Training Round / Iteration",
        yaxis_title="Average Episode Reward",
        height=450, template="plotly_dark",
        paper_bgcolor="#0a0e17", plot_bgcolor="#0d1117",
        legend=dict(orientation="h", y=-0.12),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Hyperparameter sweep results
    st.markdown("### 🔍 Hyperparameter Sweep Results (Simulated)")

    sweep_data = []
    for lr in [0.00001, 0.0001, 0.0003, 0.001, 0.003]:
        for le in [3, 5, 10]:
            for strat in ["FedAvg", "FedProx"]:
                reward = -5 + np.log10(lr + 1e-6) * 2 + le * 0.3
                reward += (2.0 if strat == "FedProx" else 0) + rng.normal(0, 1)
                sweep_data.append({
                    "Learning Rate": lr,
                    "Local Epochs": le,
                    "Strategy": strat,
                    "Final Reward": round(reward, 2),
                    "SLA Rate": round(np.clip(0.5 + reward * 0.02 + rng.normal(0, 0.03), 0, 1), 3),
                })

    sweep_df = pd.DataFrame(sweep_data)

    fig_sweep = px.scatter(
        sweep_df, x="Learning Rate", y="Final Reward",
        color="Strategy", size="Local Epochs",
        hover_data=["SLA Rate"],
        color_discrete_map={"FedAvg": "#818cf8", "FedProx": "#34d399"},
        log_x=True,
        title="Sweep: Learning Rate vs Final Reward (size = Local Epochs)",
    )
    fig_sweep.update_layout(
        height=400, template="plotly_dark",
        paper_bgcolor="#0a0e17", plot_bgcolor="#0d1117",
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    # Best runs table
    st.markdown("### 🏆 Top 5 Runs")
    top5 = sweep_df.nlargest(5, "Final Reward")
    st.dataframe(top5, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: MODEL SERVING MONITOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🚀 Model Serving Monitor":
    st.markdown("# 🚀 Model Serving — Prometheus/Grafana Style")
    st.markdown("*Real-time serving metrics (simulated)*")
    st.markdown("---")

    rng = np.random.default_rng(123)
    serving_df = simulate_serving_metrics(rng, 200)

    # Live metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Throughput", f"{serving_df['throughput_rps'].iloc[-1]:.0f} req/s",
              delta=f"{serving_df['throughput_rps'].iloc[-1] - serving_df['throughput_rps'].iloc[-2]:.1f}")
    m2.metric("P50 Latency", f"{serving_df['latency_ms'].quantile(0.5):.1f} ms")
    m3.metric("P99 Latency", f"{serving_df['latency_ms'].quantile(0.99):.1f} ms",
              delta_color="inverse")
    m4.metric("Error Rate", f"{serving_df['errors'].sum() / len(serving_df):.2%}",
              delta_color="inverse")

    # Throughput + Latency
    fig_serve = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Prediction Throughput (req/s)", "Prediction Latency (ms)"),
        vertical_spacing=0.12,
    )

    fig_serve.add_trace(
        go.Scatter(x=serving_df["timestamp"], y=serving_df["throughput_rps"],
                   fill="tozeroy", fillcolor="rgba(129, 140, 248, 0.15)",
                   line=dict(color="#818cf8", width=2), name="Throughput"),
        row=1, col=1,
    )

    fig_serve.add_trace(
        go.Scatter(x=serving_df["timestamp"], y=serving_df["latency_ms"],
                   mode="markers", marker=dict(size=3, color="#34d399"), name="Latency"),
        row=2, col=1,
    )
    fig_serve.add_hline(y=50, line_dash="dash", line_color="#f87171",
                        annotation_text="SLA: 50ms", row=2, col=1)

    fig_serve.update_layout(
        height=550, template="plotly_dark",
        paper_bgcolor="#0a0e17", plot_bgcolor="#0d1117",
        showlegend=False,
    )
    st.plotly_chart(fig_serve, use_container_width=True)

    # Latency distribution
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = go.Figure(go.Histogram(
            x=serving_df["latency_ms"], nbinsx=50,
            marker_color="#34d399",
        ))
        fig_hist.add_vline(x=serving_df["latency_ms"].quantile(0.99),
                          line_dash="dash", line_color="#f87171",
                          annotation_text="P99")
        fig_hist.update_layout(
            title="Latency Distribution",
            xaxis_title="Latency (ms)", yaxis_title="Count",
            height=300, template="plotly_dark",
            paper_bgcolor="#0a0e17", plot_bgcolor="#0d1117",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        fig_err = go.Figure(go.Bar(
            x=serving_df["timestamp"][::10],
            y=serving_df["errors"].rolling(10).sum().iloc[::10],
            marker_color="#f87171",
        ))
        fig_err.update_layout(
            title="Error Count (10-min buckets)",
            xaxis_title="Time", yaxis_title="Errors",
            height=300, template="plotly_dark",
            paper_bgcolor="#0a0e17", plot_bgcolor="#0d1117",
        )
        st.plotly_chart(fig_err, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: CI/CD PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "⚙️ CI/CD Pipeline View":
    st.markdown("# ⚙️ GitHub Actions CI/CD Pipelines")
    st.markdown("*Visual representation of the 4 automated pipelines*")
    st.markdown("---")

    pipelines = [
        {
            "name": "CI Pipeline (ci.yml)",
            "trigger": "Every PR + push to develop",
            "color": "#818cf8",
            "steps": [
                ("Checkout Code", "1s", "✅"),
                ("Setup Python 3.11", "15s", "✅"),
                ("Install Dependencies", "45s", "✅"),
                ("Ruff Lint Check", "8s", "✅"),
                ("Ruff Format Check", "5s", "✅"),
                ("MyPy Type Check", "12s", "✅"),
                ("Pytest + Coverage", "90s", "✅"),
                ("Gym Env Validation", "10s", "✅"),
                ("Upload to Codecov", "5s", "✅"),
            ]
        },
        {
            "name": "Training Pipeline (train.yml)",
            "trigger": "Merge to main + manual",
            "color": "#34d399",
            "steps": [
                ("Checkout Code", "1s", "✅"),
                ("Setup Python + Cache", "20s", "✅"),
                ("Install Dependencies", "45s", "✅"),
                ("Run Federated Training", "45min", "🔄"),
                ("Log to W&B", "—", "🔄"),
                ("Validate Model Thresholds", "30s", "⏳"),
                ("Register in MLflow (Staging)", "10s", "⏳"),
                ("Upload Artifacts", "15s", "⏳"),
            ]
        },
        {
            "name": "Deploy Pipeline (deploy.yml)",
            "trigger": "Manual with approval",
            "color": "#f97316",
            "steps": [
                ("Promote Model in MLflow", "5s", "⏳"),
                ("Build Docker Image", "120s", "⏳"),
                ("Push to Container Registry", "30s", "⏳"),
                ("Deploy to Environment", "60s", "⏳"),
                ("Health Check", "30s", "⏳"),
                ("Slack Notification", "2s", "⏳"),
            ]
        },
        {
            "name": "Release Pipeline (release.yml)",
            "trigger": "Git tag v*.*.*",
            "color": "#fbbf24",
            "steps": [
                ("Checkout (full history)", "10s", "⏳"),
                ("Generate Changelog", "5s", "⏳"),
                ("Create GitHub Release", "3s", "⏳"),
            ]
        },
    ]

    for pipeline in pipelines:
        with st.expander(f"**{pipeline['name']}** — Trigger: {pipeline['trigger']}", expanded=True):
            for i, (step_name, duration, status) in enumerate(pipeline["steps"]):
                col1, col2, col3, col4 = st.columns([0.5, 4, 1, 0.5])
                with col1:
                    st.markdown(f"**{i+1}**")
                with col2:
                    st.markdown(f"`{step_name}`")
                with col3:
                    st.markdown(f"*{duration}*")
                with col4:
                    st.markdown(status)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE: ARCHITECTURE EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🏗️ Architecture Explorer":
    st.markdown("# 🏗️ Architecture Explorer")
    st.markdown("*Click each layer to see its code and details*")
    st.markdown("---")

    layers = {
        "🌍 Custom Gym Environment": {
            "desc": "Gymnasium-compatible resource allocation simulator",
            "key_file": "src/envs/resource_alloc_env.py",
            "code": '''class ResourceAllocationEnv(gym.Env):
    def __init__(self, config=None):
        cfg = config or {}
        self.num_nodes = cfg.get("num_nodes", 5)
        self.max_pending = cfg.get("max_pending_tasks", 10)

        # Observation: node utils + task features
        obs_dim = self.num_nodes * 4 + self.max_pending * 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: assign each task to a node or reject
        self.action_space = spaces.MultiDiscrete(
            [self.num_nodes + 1] * self.max_pending
        )

    def step(self, action):
        rewards = []
        for task_idx, node_idx in enumerate(action):
            task = self.pending_tasks[task_idx]
            if node_idx == self.num_nodes:  # REJECT
                rewards.append(-3.0)       # Heavy penalty
            elif self.nodes[node_idx].can_accept:
                self._assign_task(node_idx, task)
                sla_ok = self._estimate_latency(node_idx) <= task["deadline"]
                rewards.append(1.0 + 2.0*float(sla_ok) - 0.5*latency)
        return self._get_obs(), sum(rewards), terminated, False, info''',
        },
        "🧠 PPO Agent (Ray/RLlib)": {
            "desc": "Proximal Policy Optimization configured via Ray/RLlib",
            "key_file": "src/agents/ppo_agent.py",
            "code": '''config = (
    PPOConfig()
    .environment(env="ResourceAlloc-v0", env_config=env_cfg)
    .framework("torch")
    .training(
        lr=3e-4,              # Adam optimizer learning rate
        gamma=0.99,           # Discount factor
        clip_param=0.2,       # PPO clipping for stability
        train_batch_size=4000,
        model={"fcnet_hiddens": [256, 256, 128], "fcnet_activation": "relu"},
    )
    .rollouts(num_rollout_workers=4)  # Parallel data collection
)''',
        },
        "🔄 Federated Aggregation": {
            "desc": "FedAvg / FedProx weight aggregation strategies",
            "key_file": "src/federation/strategies.py",
            "code": '''def fed_avg(global_weights, client_weights):
    """Weighted average of client model parameters."""
    total_samples = sum(n for _, n in client_weights)
    new_weights = copy.deepcopy(global_weights)

    for key in new_weights:
        new_weights[key] = torch.zeros_like(new_weights[key])
        for client_w, n_samples in client_weights:
            weight = n_samples / total_samples
            new_weights[key] += client_w[key] * weight

    return new_weights

# FedProx adds this to LOCAL training loss:
# loss += (mu/2) * ||w_local - w_global||^2''',
        },
        "📡 Federation Server + Clients": {
            "desc": "Ray Actors for distributed server-client communication",
            "key_file": "src/federation/server.py",
            "code": '''@ray.remote
class FederationServer:
    def run_federation(self):
        for round_num in range(self.num_rounds):
            # 1. Send global weights to all clients
            global_weights = self.global_model.state_dict()
            ray.get([c.set_weights.remote(global_weights)
                     for c in self.clients])

            # 2. Parallel local training
            results = ray.get([c.train_local.remote(self.local_epochs)
                              for c in self.clients])

            # 3. Aggregate
            weight_pairs = [(w, n) for w, n, _ in results]
            new_global = fed_avg(global_weights, weight_pairs)
            self.global_model.load_state_dict(new_global)

            # 4. Log metrics to W&B + Prometheus
            wandb.log({"global/avg_reward": avg_reward, ...})''',
        },
        "📊 W&B + MLflow": {
            "desc": "Experiment tracking + model versioning",
            "key_file": "src/mlops/experiment_tracker.py + model_registry.py",
            "code": '''# W&B: Live tracking during training
wandb.init(project="federated-rl-resource-alloc",
           config=config, tags=["fedavg"])
wandb.log({"round": r, "global/avg_reward": reward,
           "client_0/sla_rate": 0.87})

# MLflow: Model versioning after training
with mlflow.start_run(run_name="gha-abc123"):
    mlflow.log_params({"strategy": "fedavg", "clients": 5})
    mlflow.log_metric("avg_reward", 42.5)
    mlflow.pytorch.log_model(model, "model",
        registered_model_name="fed-rl-resource-alloc")
# Promote: client.transition_model_version_stage(
#     name="fed-rl-resource-alloc", version=3, stage="Production")''',
        },
        "🔔 Prometheus + Grafana": {
            "desc": "Metrics collection, time-series storage, dashboards, alerts",
            "key_file": "src/mlops/metrics_exporter.py",
            "code": '''from prometheus_client import Counter, Gauge, Histogram, start_http_server

round_counter = Counter("fed_rounds_total", "Rounds completed")
global_reward = Gauge("fed_global_reward", "Current reward")
pred_latency  = Histogram("prediction_latency_seconds",
                          "Prediction latency",
                          buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0])

start_http_server(8000)  # Prometheus scrapes this

# After each round:
round_counter.inc()
global_reward.set(42.5)

# Alert rule (prometheus.yml):
# - alert: HighPredictionLatency
#   expr: histogram_quantile(0.99, prediction_latency_seconds_bucket) > 1.0
#   for: 2m
#   severity: critical''',
        },
        "🚀 FastAPI Serving": {
            "desc": "Production model serving with Prometheus metrics",
            "key_file": "src/mlops/serving.py",
            "code": '''app = FastAPI(title="Fed-RL Resource Allocator")
MODEL = mlflow.pytorch.load_model("models:/fed-rl-resource-alloc/Production")

@app.post("/predict")
async def predict(req: PredictRequest):
    start = time.time()
    obs = torch.FloatTensor(req.observation).unsqueeze(0)
    with torch.no_grad():
        action = MODEL(obs).argmax(dim=-1)
    PREDICT_LATENCY.observe(time.time() - start)
    PREDICT_COUNT.inc()
    return {"action": action.tolist()}

@app.get("/metrics")  # Prometheus scrapes this
async def metrics():
    return Response(generate_latest(), media_type="text/plain")''',
        },
    }

    for layer_name, info in layers.items():
        with st.expander(f"**{layer_name}** — {info['desc']}", expanded=False):
            st.markdown(f"📁 **Key File:** `{info['key_file']}`")
            st.code(info["code"], language="python")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; padding:20px;'>"
    "Federated RL Resource Allocation — Visualization Dashboard<br/>"
    "<small>Built with Streamlit + Plotly | Simulated data for demonstration</small>"
    "</div>",
    unsafe_allow_html=True,
)