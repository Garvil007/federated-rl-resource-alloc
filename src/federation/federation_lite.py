import asyncio
import json
import time
import base64
import struct
import argparse
import platform
import os
import sys
import logging
from typing import Dict, List, Tuple
from collections import OrderedDict

import numpy as np

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [LITE] %(message)s")
logger = logging.getLogger("LiteClient")


# ═══════════════════════════════════════════════════════
#  NUMPY NEURAL NETWORK (replaces PyTorch)
# ═══════════════════════════════════════════════════════

class NumpyLinear:
    """A single linear layer: y = x @ W^T + b"""
    def __init__(self, in_features: int, out_features: int):
        # Xavier initialization
        scale = np.sqrt(2.0 / in_features)
        self.weight = np.random.randn(out_features, in_features).astype(np.float32) * scale
        self.bias = np.zeros(out_features, dtype=np.float32)
        # Gradients
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)
        # Cache for backward pass
        self._input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return x @ self.weight.T + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.weight_grad = grad_output.T @ self._input
        self.bias_grad = grad_output.sum(axis=0)
        return grad_output @ self.weight


class NumpyReLU:
    """ReLU activation: max(0, x)"""
    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0).astype(np.float32)
        return x * self._mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self._mask


class NumpyPolicyNetwork:
    """
    Same architecture as the PyTorch PolicyNetwork:
    obs_dim → 256 → ReLU → 256 → ReLU → 128 → ReLU → act_dim

    Weights are stored in the same format as PyTorch's state_dict
    so they can be directly loaded from/saved to the server.
    """
    def __init__(self, obs_dim: int = 50, act_dim: int = 6):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.layers = [
            NumpyLinear(obs_dim, 256), NumpyReLU(),
            NumpyLinear(256, 256),     NumpyReLU(),
            NumpyLinear(256, 128),     NumpyReLU(),
            NumpyLinear(128, act_dim),
        ]
        self.linear_layers = [l for l in self.layers if isinstance(l, NumpyLinear)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    def get_action(self, obs: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """Sample action from policy."""
        logits = self.forward(obs.reshape(1, -1))
        probs = self.softmax(logits).squeeze()
        action = np.random.choice(len(probs), p=probs)
        return action, probs

    def load_pytorch_state_dict(self, raw_bytes: bytes):
        """
        Load weights from a PyTorch state_dict (serialized as bytes).

        PyTorch's state_dict keys for nn.Sequential are:
            net.0.weight, net.0.bias, net.2.weight, net.2.bias, etc.
        (indices 0,2,4,6 are Linear layers; 1,3,5 are ReLU)

        We parse the raw bytes to extract the tensor data.
        """
        try:
            import torch
            import io
            buffer = io.BytesIO(raw_bytes)
            state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
            linear_idx = 0
            for key in sorted(state_dict.keys()):
                tensor = state_dict[key].numpy()
                if "weight" in key:
                    self.linear_layers[linear_idx].weight = tensor.astype(np.float32)
                elif "bias" in key:
                    self.linear_layers[linear_idx].bias = tensor.astype(np.float32)
                    linear_idx += 1
        except ImportError:
            logger.warning("PyTorch not available — using manual weight parsing")
            self._parse_weights_manual(raw_bytes)

    def _parse_weights_manual(self, raw_bytes: bytes):
        """Fallback: parse PyTorch saved weights without importing torch."""
        # This is a simplified parser. For production, use a proper
        # format like safetensors or ONNX.
        logger.warning("Manual weight parsing — weights may not load correctly without PyTorch")

    def save_pytorch_state_dict(self) -> bytes:
        """Convert numpy weights back to PyTorch state_dict format."""
        try:
            import torch
            import io
            state_dict = OrderedDict()
            for i, layer in enumerate(self.linear_layers):
                # PyTorch Sequential indices: 0, 2, 4, 6 (skipping ReLU at 1, 3, 5)
                pt_idx = i * 2
                state_dict[f"net.{pt_idx}.weight"] = torch.tensor(layer.weight)
                state_dict[f"net.{pt_idx}.bias"] = torch.tensor(layer.bias)
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            return buffer.getvalue()
        except ImportError:
            logger.error("PyTorch needed to serialize weights back to server format")
            # Fallback: send as numpy arrays in a custom format
            return self._serialize_numpy_weights()

    def _serialize_numpy_weights(self) -> bytes:
        """Serialize weights as raw numpy bytes (fallback)."""
        import pickle
        data = {}
        for i, layer in enumerate(self.linear_layers):
            data[f"layer_{i}_weight"] = layer.weight
            data[f"layer_{i}_bias"] = layer.bias
        return pickle.dumps(data)

    def update_weights(self, lr: float = 0.0003):
        """Apply accumulated gradients (simple SGD)."""
        for layer in self.linear_layers:
            layer.weight -= lr * layer.weight_grad
            layer.bias -= lr * layer.bias_grad
            layer.weight_grad = np.zeros_like(layer.weight)
            layer.bias_grad = np.zeros_like(layer.bias)


# ═══════════════════════════════════════════════════════
#  LOCAL ENVIRONMENT (same as full client, numpy only)
# ═══════════════════════════════════════════════════════

class LiteEnvironment:
    """Lightweight environment using only numpy."""

    PROFILES = {
        "android": {"num_nodes": 2, "arrival": 1.0, "ep_len": 80},
        "ios":     {"num_nodes": 2, "arrival": 1.2, "ep_len": 80},
        "windows": {"num_nodes": 4, "arrival": 3.0, "ep_len": 120},
        "macos":   {"num_nodes": 4, "arrival": 2.5, "ep_len": 120},
        "linux":   {"num_nodes": 6, "arrival": 5.0, "ep_len": 150},
    }

    def __init__(self, device_os="linux", obs_dim=50, act_dim=6):
        p = self.PROFILES.get(device_os, self.PROFILES["linux"])
        self.num_nodes = p["num_nodes"]
        self.arrival = p["arrival"]
        self.ep_len = p["ep_len"]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_pending = (obs_dim - self.num_nodes * 4) // 3
        self.rng = np.random.default_rng(int(time.time()) % 2**31)
        self.reset()

    def reset(self):
        self.node_cpu = self.rng.uniform(0.1, 0.5, self.num_nodes)
        self.node_mem = self.rng.uniform(0.1, 0.4, self.num_nodes)
        self.node_queue = np.zeros(self.num_nodes, dtype=int)
        self.step_count = 0
        self.completed = 0
        self.dropped = 0
        self.sla_met = 0
        self.tasks = self._gen_tasks()
        return self._obs()

    def step(self, action_idx: int):
        n_tasks = len(self.tasks)
        reward = 0.0

        if n_tasks > 0:
            if action_idx < self.num_nodes and self.node_cpu[action_idx] < 0.95:
                self.node_cpu[action_idx] = min(self.node_cpu[action_idx] + 0.1, 0.95)
                self.node_queue[action_idx] += 1
                latency = self.node_queue[action_idx] * 0.1 + self.node_cpu[action_idx] * 2
                sla_ok = latency <= self.tasks[0]["deadline"]
                self.completed += 1
                if sla_ok: self.sla_met += 1
                reward = 1.0 + 2.0 * float(sla_ok) - 0.5 * latency - 0.1
            else:
                self.dropped += 1
                reward = -3.0

        # Decay
        self.node_cpu *= 0.95
        self.node_mem *= 0.95
        self.node_queue = np.maximum(0, self.node_queue - self.rng.integers(0, 2, self.num_nodes))

        self.step_count += 1
        self.tasks = self._gen_tasks()
        done = self.step_count >= self.ep_len

        return self._obs(), reward, done

    def _obs(self):
        feats = []
        for i in range(self.num_nodes):
            feats.extend([self.node_cpu[i], self.node_mem[i], 0.5, self.node_queue[i]/20])
        for i in range(self.max_pending):
            if i < len(self.tasks):
                t = self.tasks[i]
                feats.extend([t["cpu"]/100, t["mem"]/64, t["deadline"]/10])
            else:
                feats.extend([0, 0, 0])
        feats = feats[:self.obs_dim] + [0]*(self.obs_dim - len(feats[:self.obs_dim]))
        return np.array(feats, dtype=np.float32)

    def _gen_tasks(self):
        n = min(int(self.rng.poisson(self.arrival)), max(1, self.max_pending))
        return [{"cpu": float(self.rng.uniform(5,50)), "mem": float(self.rng.uniform(1,32)),
                 "deadline": float(self.rng.uniform(1,10))} for _ in range(n)]


# ═══════════════════════════════════════════════════════
#  LOCAL TRAINING (numpy-only, simplified REINFORCE)
# ═══════════════════════════════════════════════════════

def train_locally_lite(policy, env, num_epochs, lr):
    total_reward = 0.0
    total_samples = 0

    for epoch in range(num_epochs):
        obs = env.reset()
        ep_reward = 0.0

        for step in range(env.ep_len):
            action, probs = policy.get_action(obs)
            obs_next, reward, done = env.step(action)
            ep_reward += reward
            total_samples += 1
            obs = obs_next
            if done:
                break

        total_reward += ep_reward

        # Simple weight perturbation update (no backprop needed)
        # This is a form of evolutionary strategy / random search
        for layer in policy.linear_layers:
            noise_w = np.random.randn(*layer.weight.shape).astype(np.float32) * 0.01
            noise_b = np.random.randn(*layer.bias.shape).astype(np.float32) * 0.01
            if ep_reward > 0:
                layer.weight += lr * noise_w * ep_reward * 0.01
                layer.bias += lr * noise_b * ep_reward * 0.01
            else:
                layer.weight -= lr * noise_w * abs(ep_reward) * 0.001
                layer.bias -= lr * noise_b * abs(ep_reward) * 0.001

    sla_rate = env.sla_met / max(env.completed, 1)
    return {
        "reward": total_reward / num_epochs,
        "sla_rate": sla_rate,
        "samples": total_samples,
        "completed": env.completed,
        "dropped": env.dropped,
    }


# ═══════════════════════════════════════════════════════
#  WEBSOCKET CLIENT
# ═══════════════════════════════════════════════════════

async def run_lite_client(server_url, device_name, device_os):
    logger.info(f"Connecting to {server_url}...")

    policy = NumpyPolicyNetwork(obs_dim=50, act_dim=6)
    env = LiteEnvironment(device_os=device_os)

    async with websockets.connect(server_url, max_size=500*1024*1024, ping_interval=30) as ws:
        # Register
        await ws.send(json.dumps({
            "type": "register", "device_name": device_name,
            "device_os": device_os,
        }))
        raw = await ws.recv()
        msg = json.loads(raw)
        if msg["type"] != "registered":
            logger.error(f"Registration failed: {msg}")
            return

        config = msg["config"]
        policy = NumpyPolicyNetwork(config["obs_dim"], config["act_dim"])
        env = LiteEnvironment(device_os, config["obs_dim"], config["act_dim"])
        logger.info(f"✅ Registered! Waiting for training commands...")

        while True:
            try:
                raw = await ws.recv()
                msg = json.loads(raw)

                if msg["type"] == "start_training":
                    round_num = msg["round"]
                    logger.info(f"📥 Round {round_num+1} — loading weights")

                    weight_bytes = base64.b64decode(msg["weights"])
                    policy.load_pytorch_state_dict(weight_bytes)

                    logger.info(f"🏋️ Training locally...")
                    t0 = time.time()
                    metrics = train_locally_lite(
                        policy, env,
                        num_epochs=msg.get("local_epochs", 5),
                        lr=msg.get("learning_rate", 0.0003),
                    )
                    dt = time.time() - t0
                    logger.info(f"✅ Done in {dt:.1f}s | Reward: {metrics['reward']:.2f}")

                    weights_bytes = policy.save_pytorch_state_dict()
                    await ws.send(json.dumps({
                        "type": "training_complete",
                        "round": round_num,
                        "weights": base64.b64encode(weights_bytes).decode(),
                        "num_samples": metrics["samples"],
                        "metrics": metrics,
                    }))
                    logger.info("📤 Sent update")

                elif msg["type"] == "shutdown":
                    break

            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection lost")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated RL Lite Client (NumPy only)")
    parser.add_argument("--server", default="ws://localhost:8765/ws/federate")
    parser.add_argument("--name", default=f"{platform.node()}-lite")
    parser.add_argument("--os", default="android", choices=["android","ios","windows","linux","macos"])
    args = parser.parse_args()

    print(f"Lite Client: {args.name} ({args.os}) → {args.server}")
    asyncio.run(run_lite_client(args.server, args.name, args.os))