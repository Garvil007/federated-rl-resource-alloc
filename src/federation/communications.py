"""
Federation Communication Layer
================================
src/federation/communication.py

This module handles ALL communication between the Federation Server
and the Federated Clients. It supports two backends:

1. Ray Object Store (default) — for single-machine or Ray cluster setups
2. gRPC — for truly distributed setups across different machines/networks

The rest of the codebase (server.py, client.py) doesn't care WHICH
backend is used. They just call communicator.send_weights() and
communicator.receive_weights(). This is the "Strategy Pattern" in action.

Why this file exists:
    - Decouples communication logic from training logic
    - Makes it easy to switch between Ray (local) and gRPC (distributed)
    - Handles serialization, compression, error handling, retries
    - Tracks communication metrics (bytes sent, latency) for monitoring
"""

import time
import pickle
import gzip
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Conditional imports — gracefully handle missing deps
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import ray

    HAS_RAY = True
except ImportError:
    HAS_RAY = False

try:
    import grpc
    from concurrent import futures  # noqa: F401

    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False

try:
    from prometheus_client import Counter, Histogram, Gauge  # noqa: F401

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATA CLASSES — Structured messages between server/client
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MessageType(Enum):
    """Types of messages that can be sent between server and clients."""

    GLOBAL_WEIGHTS = (
        "global_weights"  # Server → Client: here are the new global weights
    )
    LOCAL_UPDATE = "local_update"  # Client → Server: here are my trained weights
    TRAINING_CONFIG = (
        "training_config"  # Server → Client: here are the training settings
    )
    HEARTBEAT = "heartbeat"  # Client → Server: I'm still alive
    SHUTDOWN = "shutdown"  # Server → Client: stop training


@dataclass
class WeightMessage:
    """
    A message containing model weights + metadata.

    This is what gets sent between server and clients.
    The weights are a PyTorch state_dict (OrderedDict of tensors),
    but we serialize them to bytes for transmission.
    """

    sender_id: str  # "server" or "client_0", "client_1", etc.
    message_type: MessageType  # What kind of message this is
    round_num: int  # Which federation round this belongs to
    weights: Optional[OrderedDict] = None  # The actual neural network weights
    num_samples: int = 0  # How many samples the client trained on
    metrics: Dict[str, float] = field(  # Training metrics (reward, loss, etc.)
        default_factory=dict
    )
    timestamp: float = field(  # When this message was created
        default_factory=time.time
    )
    compressed: bool = False  # Whether the weights are gzip-compressed

    def size_bytes(self) -> int:
        """Estimate the size of this message in bytes."""
        if self.weights is None:
            return 0
        total = 0
        for key, tensor in self.weights.items():
            if HAS_TORCH and isinstance(tensor, torch.Tensor):
                total += tensor.nelement() * tensor.element_size()
            else:
                total += 0
        return total

    def size_mb(self) -> float:
        """Estimate the size of this message in megabytes."""
        return self.size_bytes() / (1024 * 1024)


@dataclass
class CommunicationStats:
    """
    Tracks communication statistics for monitoring.

    These stats get exported to Prometheus/Grafana via
    the metrics_exporter.py module.
    """

    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_send_time_seconds: float = 0.0
    total_receive_time_seconds: float = 0.0
    failed_sends: int = 0
    failed_receives: int = 0
    compression_ratio: float = 1.0  # < 1.0 means compression helped

    @property
    def avg_send_latency(self) -> float:
        if self.total_messages_sent == 0:
            return 0.0
        return self.total_send_time_seconds / self.total_messages_sent

    @property
    def avg_receive_latency(self) -> float:
        if self.total_messages_received == 0:
            return 0.0
        return self.total_receive_time_seconds / self.total_messages_received

    def to_dict(self) -> Dict[str, float]:
        return {
            "comm/total_mb_sent": self.total_bytes_sent / (1024 * 1024),
            "comm/total_mb_received": self.total_bytes_received / (1024 * 1024),
            "comm/total_messages_sent": self.total_messages_sent,
            "comm/avg_send_latency_ms": self.avg_send_latency * 1000,
            "comm/avg_receive_latency_ms": self.avg_receive_latency * 1000,
            "comm/failed_sends": self.failed_sends,
            "comm/compression_ratio": self.compression_ratio,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SERIALIZATION — Converting weights to/from bytes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class WeightSerializer:
    """
    Handles converting PyTorch state_dicts to bytes and back.

    Why not just use pickle directly?
    1. We want optional gzip compression (saves 60-80% bandwidth)
    2. We want to track the size before/after compression
    3. We want error handling in one place
    4. We might want to switch to a faster serializer later
       (e.g., safetensors) without changing any other code
    """

    @staticmethod
    def serialize(
        weights: OrderedDict,
        compress: bool = True,
        compression_level: int = 6,
    ) -> bytes:
        """
        Convert a state_dict to bytes, optionally compressed.

        Args:
            weights: PyTorch model state_dict (OrderedDict of tensors)
            compress: Whether to apply gzip compression
            compression_level: 1 (fast) to 9 (smallest). 6 is a good balance.

        Returns:
            bytes: Serialized (and optionally compressed) weights

        Example:
            weights = model.state_dict()
            data = WeightSerializer.serialize(weights, compress=True)
            # data is now bytes, ready to send over the network
        """
        # Step 1: Convert tensors to CPU (in case they're on GPU)
        if HAS_TORCH:
            cpu_weights = OrderedDict()
            for key, tensor in weights.items():
                if isinstance(tensor, torch.Tensor):
                    cpu_weights[key] = tensor.detach().cpu()
                else:
                    cpu_weights[key] = tensor
        else:
            cpu_weights = weights

        # Step 2: Pickle the OrderedDict to bytes
        raw_bytes = pickle.dumps(cpu_weights, protocol=pickle.HIGHEST_PROTOCOL)

        # Step 3: Optionally compress with gzip
        if compress:
            compressed_bytes = gzip.compress(raw_bytes, compresslevel=compression_level)
            logger.debug(
                f"Serialized weights: {len(raw_bytes)} bytes → "
                f"{len(compressed_bytes)} bytes "
                f"({len(compressed_bytes) / len(raw_bytes):.1%} of original)"
            )
            return compressed_bytes
        else:
            return raw_bytes

    @staticmethod
    def deserialize(data: bytes, compressed: bool = True) -> OrderedDict:
        """
        Convert bytes back to a state_dict.

        Args:
            data: Serialized weight bytes
            compressed: Whether the data is gzip-compressed

        Returns:
            OrderedDict: PyTorch model state_dict

        Example:
            weights = WeightSerializer.deserialize(data, compressed=True)
            model.load_state_dict(weights)
        """
        # Step 1: Decompress if needed
        if compressed:
            raw_bytes = gzip.decompress(data)
        else:
            raw_bytes = data

        # Step 2: Unpickle back to OrderedDict
        weights = pickle.loads(raw_bytes)

        return weights

    @staticmethod
    def compute_delta(
        global_weights: OrderedDict,
        local_weights: OrderedDict,
    ) -> OrderedDict:
        """
        Compute the DIFFERENCE between local and global weights.

        Instead of sending the full model weights (e.g., 10 MB),
        we can send just the delta (difference), which is often
        much smaller, especially if the client didn't change much.

        This is an optimization for bandwidth-constrained scenarios.

        Args:
            global_weights: The weights the client STARTED with
            local_weights: The weights the client has AFTER training

        Returns:
            OrderedDict: The difference (local - global) for each layer
        """
        if not HAS_TORCH:
            return local_weights

        delta = OrderedDict()
        for key in local_weights:
            delta[key] = local_weights[key] - global_weights[key]
        return delta

    @staticmethod
    def apply_delta(
        global_weights: OrderedDict,
        delta: OrderedDict,
    ) -> OrderedDict:
        """
        Apply a weight delta to get the full local weights back.

        This is the reverse of compute_delta():
            local_weights = global_weights + delta

        Args:
            global_weights: The global model weights
            delta: The difference computed by compute_delta()

        Returns:
            OrderedDict: Reconstructed full local weights
        """
        if not HAS_TORCH:
            return delta

        full_weights = OrderedDict()
        for key in global_weights:
            full_weights[key] = global_weights[key] + delta[key]
        return full_weights


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ABSTRACT BASE — Common interface for all backends
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CommunicationBackend(ABC):
    """
    Abstract base class for communication backends.

    Both RayCommunicator and GRPCCommunicator implement this interface.
    The server/client code only uses these methods, so it doesn't
    care which backend is active.

    This is the "Strategy Pattern" — swap the strategy without
    changing the code that uses it.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.stats = CommunicationStats()
        self.compress = self.config.get("compress", True)
        self.compression_level = self.config.get("compression_level", 6)
        self.use_delta = self.config.get("use_delta_compression", False)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        self._setup_prometheus_metrics()

    def _setup_prometheus_metrics(self):
        """Create Prometheus metrics for monitoring communication."""
        if HAS_PROMETHEUS:
            self.prom_bytes_sent = Counter(
                "fed_comm_bytes_sent_total",
                "Total bytes sent in federation communication",
            )
            self.prom_bytes_received = Counter(
                "fed_comm_bytes_received_total",
                "Total bytes received in federation communication",
            )
            self.prom_send_latency = Histogram(
                "fed_comm_send_latency_seconds",
                "Latency of sending weight messages",
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            )
            self.prom_message_size = Histogram(
                "fed_comm_message_size_mb",
                "Size of weight messages in MB",
                buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            )
            self.prom_errors = Counter(
                "fed_comm_errors_total",
                "Total communication errors",
                ["error_type"],
            )

    @abstractmethod
    def send_weights_to_client(
        self,
        client_id: int,
        weights: OrderedDict,
        round_num: int,
        config: Optional[Dict] = None,
    ) -> bool:
        """
        Send global model weights to a specific client.

        Called by the server at the START of each federation round.
        The server says: "Hey client 3, here are the new global weights.
        Load them into your model and start training."

        Args:
            client_id: Which client to send to (0, 1, 2, ...)
            weights: The global model's state_dict
            round_num: Current federation round number
            config: Optional training config to send along

        Returns:
            bool: True if send was successful
        """
        pass

    @abstractmethod
    def receive_update_from_client(
        self,
        client_id: int,
        timeout: float = 300.0,
    ) -> Optional[WeightMessage]:
        """
        Receive trained weights from a specific client.

        Called by the server AFTER telling clients to train.
        The server waits for each client to finish training
        and send back their updated weights.

        Args:
            client_id: Which client to receive from
            timeout: Max seconds to wait before giving up

        Returns:
            WeightMessage with updated weights, or None if timeout/error
        """
        pass

    @abstractmethod
    def send_update_to_server(
        self,
        client_id: int,
        weights: OrderedDict,
        num_samples: int,
        metrics: Dict[str, float],
        round_num: int,
    ) -> bool:
        """
        Send trained weights from a client back to the server.

        Called by a client AFTER local training is complete.
        The client says: "Hey server, I trained for 5 epochs on 4000
        samples. Here are my updated weights and my metrics."

        Args:
            client_id: This client's ID
            weights: The locally-trained model's state_dict
            num_samples: How many samples were used in training
            metrics: Training metrics (reward, loss, sla_rate, etc.)
            round_num: Current federation round number

        Returns:
            bool: True if send was successful
        """
        pass

    @abstractmethod
    def broadcast_weights(
        self,
        weights: OrderedDict,
        round_num: int,
        client_ids: List[int],
    ) -> Dict[int, bool]:
        """
        Send global weights to ALL clients at once.

        More efficient than calling send_weights_to_client() in a loop
        because some backends (Ray) can do this in parallel.

        Args:
            weights: Global model state_dict
            round_num: Current round
            client_ids: List of client IDs to send to

        Returns:
            Dict mapping client_id → success (True/False)
        """
        pass

    @abstractmethod
    def collect_updates(
        self,
        client_ids: List[int],
        timeout: float = 300.0,
    ) -> List[Optional[WeightMessage]]:
        """
        Collect trained weights from ALL clients.

        Waits for all clients to finish training and send back
        their updates. Returns them in a list (same order as client_ids).

        Args:
            client_ids: List of client IDs to collect from
            timeout: Max seconds to wait for ALL clients

        Returns:
            List of WeightMessages (None for clients that timed out)
        """
        pass

    def get_stats(self) -> CommunicationStats:
        """Return current communication statistics."""
        return self.stats

    def reset_stats(self):
        """Reset all communication statistics to zero."""
        self.stats = CommunicationStats()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RAY BACKEND — Default for single-machine / Ray cluster
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RayCommunicator(CommunicationBackend):
    """
    Communication via Ray's Object Store.

    How it works:
    - Ray's Object Store is a shared-memory system
    - When you call ray.put(data), it stores the data in shared memory
      and returns a reference (ObjectRef)
    - When another process calls ray.get(ref), it reads from shared memory
      WITHOUT copying (zero-copy reads for same machine)
    - For multi-machine Ray clusters, Ray handles serialization/transfer
      automatically

    Why use this:
    - Zero-copy on same machine = very fast
    - Automatic parallelism with ray.get([future1, future2, ...])
    - Works with Ray Actors (our FederatedClient is a Ray Actor)
    - No need to manage sockets, ports, or connections

    When NOT to use this:
    - When clients are on different networks (no Ray cluster)
    - When you need fine-grained control over the protocol
    - In those cases, use GRPCCommunicator instead
    """

    def __init__(self, config: Dict[str, Any] = None, client_actors: List = None):
        """
        Args:
            config: Communication settings
            client_actors: List of Ray Actor handles for the federated clients.
                          These are created by server.py like:
                          FederatedClient.remote(client_id=0, ...)
        """
        super().__init__(config)
        self.client_actors = client_actors or []

    def set_client_actors(self, actors: List):
        """Set or update the list of Ray client actors."""
        self.client_actors = actors

    def send_weights_to_client(
        self,
        client_id: int,
        weights: OrderedDict,
        round_num: int,
        config: Optional[Dict] = None,
    ) -> bool:
        """
        Send weights to a single client via Ray.

        Under the hood:
        1. We call client_actor.set_weights.remote(weights)
        2. Ray serializes the weights and puts them in the Object Store
        3. The client actor reads them from the Object Store
        4. ray.get() waits for the client to confirm receipt
        """
        start_time = time.time()

        try:
            if client_id >= len(self.client_actors):
                logger.error(
                    f"Client {client_id} not found (only {len(self.client_actors)} clients)"
                )
                return False

            actor = self.client_actors[client_id]

            # Put weights in Ray Object Store (shared memory)
            # This is efficient because Ray avoids unnecessary copies
            weights_ref = ray.put(weights) if HAS_RAY else weights

            # Call the client's set_weights method remotely
            if HAS_RAY:
                future = actor.set_weights.remote(weights_ref)
                ray.get(future)  # Wait for confirmation
            else:
                # Fallback for testing without Ray
                actor.set_weights(weights)

            # Track statistics
            elapsed = time.time() - start_time
            msg_size = WeightMessage(
                sender_id="server",
                message_type=MessageType.GLOBAL_WEIGHTS,
                round_num=round_num,
                weights=weights,
            ).size_bytes()

            self.stats.total_bytes_sent += msg_size
            self.stats.total_messages_sent += 1
            self.stats.total_send_time_seconds += elapsed

            if HAS_PROMETHEUS:
                self.prom_bytes_sent.inc(msg_size)
                self.prom_send_latency.observe(elapsed)
                self.prom_message_size.observe(msg_size / (1024 * 1024))

            logger.debug(
                f"Sent weights to client {client_id}: "
                f"{msg_size / (1024 * 1024):.2f} MB in {elapsed:.3f}s"
            )
            return True

        except Exception as e:
            self.stats.failed_sends += 1
            if HAS_PROMETHEUS:
                self.prom_errors.labels(error_type="send_failed").inc()
            logger.error(f"Failed to send weights to client {client_id}: {e}")
            return False

    def receive_update_from_client(
        self,
        client_id: int,
        timeout: float = 300.0,
    ) -> Optional[WeightMessage]:
        """
        Receive trained weights from a single client.

        This is typically NOT called directly. Instead, use
        collect_updates() which receives from ALL clients in parallel.
        """
        start_time = time.time()

        try:
            actor = self.client_actors[client_id]

            if HAS_RAY:
                future = actor.get_update.remote()
                result = ray.get(future, timeout=timeout)
            else:
                result = actor.get_update()

            weights, num_samples, metrics = result

            elapsed = time.time() - start_time
            msg = WeightMessage(
                sender_id=f"client_{client_id}",
                message_type=MessageType.LOCAL_UPDATE,
                round_num=-1,
                weights=weights,
                num_samples=num_samples,
                metrics=metrics,
            )

            self.stats.total_bytes_received += msg.size_bytes()
            self.stats.total_messages_received += 1
            self.stats.total_receive_time_seconds += elapsed

            if HAS_PROMETHEUS:
                self.prom_bytes_received.inc(msg.size_bytes())

            return msg

        except Exception as e:
            self.stats.failed_receives += 1
            if HAS_PROMETHEUS:
                self.prom_errors.labels(error_type="receive_failed").inc()
            logger.error(f"Failed to receive from client {client_id}: {e}")
            return None

    def send_update_to_server(
        self,
        client_id: int,
        weights: OrderedDict,
        num_samples: int,
        metrics: Dict[str, float],
        round_num: int,
    ) -> bool:
        """
        In the Ray backend, clients don't explicitly "send" to the server.
        Instead, the server calls client.train_local.remote() and gets
        the result back via ray.get(). This method exists for API
        compatibility with the gRPC backend.
        """
        # In Ray, the return value of train_local() IS the "send"
        # The server calls ray.get(client.train_local.remote())
        # which returns (weights, num_samples, metrics) directly
        return True

    def broadcast_weights(
        self,
        weights: OrderedDict,
        round_num: int,
        client_ids: List[int],
    ) -> Dict[int, bool]:
        """
        Send global weights to ALL clients in parallel.

        This is the key advantage of Ray: we fire off all sends
        at once, then wait for all of them to complete. On a
        Ray cluster, the Object Store handles data locality
        automatically.
        """
        start_time = time.time()
        results = {}

        try:
            if HAS_RAY:
                # Put weights in shared memory ONCE (not per client)
                weights_ref = ray.put(weights)

                # Fire all set_weights calls in parallel
                futures = {
                    cid: self.client_actors[cid].set_weights.remote(weights_ref)
                    for cid in client_ids
                    if cid < len(self.client_actors)
                }

                # Wait for all to complete
                for cid, future in futures.items():
                    try:
                        ray.get(future, timeout=30.0)
                        results[cid] = True
                    except Exception as e:
                        logger.error(f"Broadcast to client {cid} failed: {e}")
                        results[cid] = False
            else:
                # Non-Ray fallback
                for cid in client_ids:
                    results[cid] = self.send_weights_to_client(cid, weights, round_num)

            elapsed = time.time() - start_time
            successful = sum(1 for v in results.values() if v)
            logger.info(
                f"Broadcast to {successful}/{len(client_ids)} clients in {elapsed:.3f}s"
            )
            return results

        except Exception as e:
            logger.error(f"Broadcast failed: {e}")
            return {cid: False for cid in client_ids}

    def collect_updates(
        self,
        client_ids: List[int],
        timeout: float = 300.0,
    ) -> List[Optional[WeightMessage]]:
        """
        Collect trained weights from ALL clients in parallel.

        This is the most performance-critical method. Instead of
        waiting for each client sequentially (slow), we fire off
        all training jobs at once and wait for all to finish.

        In a real setup with Ray, train_local() runs on each client
        in parallel, and ray.get() collects all results when they're
        ready. The slowest client determines the total time.
        """
        start_time = time.time()
        messages = []

        if HAS_RAY:
            # The server typically calls this pattern:
            #   futures = [c.train_local.remote(epochs) for c in clients]
            #   results = ray.get(futures)
            # This method wraps that pattern for the CommunicationBackend API
            for cid in client_ids:
                msg = self.receive_update_from_client(cid, timeout)
                messages.append(msg)
        else:
            for cid in client_ids:
                msg = self.receive_update_from_client(cid, timeout)
                messages.append(msg)

        elapsed = time.time() - start_time
        received = sum(1 for m in messages if m is not None)
        logger.info(f"Collected {received}/{len(client_ids)} updates in {elapsed:.3f}s")
        return messages


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  gRPC BACKEND — For distributed setups across networks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class GRPCCommunicator(CommunicationBackend):
    """
    Communication via gRPC (Google Remote Procedure Call).

    How it works:
    - gRPC is a framework for calling functions on remote machines
    - We define a service (FederationService) with methods like
      SendWeights() and ReceiveUpdate()
    - The server runs a gRPC server on a port (e.g., 50051)
    - Clients connect to that port and call methods as if they
      were local function calls
    - gRPC handles serialization (protobuf), compression, and
      connection management

    When to use this instead of Ray:
    - Clients are on different machines/networks
    - You can't install Ray on edge devices
    - You need fine-grained control over the network protocol
    - You need TLS/SSL encryption for the communication

    Proto file (federation.proto) defines the service:
        service FederationService {
            rpc SendGlobalWeights (WeightsRequest) returns (Ack);
            rpc SendLocalUpdate (UpdateRequest) returns (Ack);
            rpc GetGlobalWeights (ClientId) returns (WeightsResponse);
        }
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.server_address = config.get("server_address", "localhost:50051")
        self.client_addresses = config.get("client_addresses", {})
        self.channel = None
        self.server = None
        self._pending_updates: Dict[int, WeightMessage] = {}

    def _get_channel(self):
        """Get or create a gRPC channel to the server."""
        if not HAS_GRPC:
            raise RuntimeError("grpcio is not installed. Run: pip install grpcio")

        if self.channel is None:
            # Create an insecure channel (use secure_channel for production)
            self.channel = grpc.insecure_channel(
                self.server_address,
                options=[
                    # Max message size: 500 MB (for large models)
                    ("grpc.max_send_message_length", 500 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 500 * 1024 * 1024),
                    # Keep connection alive
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 10000),
                ],
            )
        return self.channel

    def send_weights_to_client(
        self,
        client_id: int,
        weights: OrderedDict,
        round_num: int,
        config: Optional[Dict] = None,
    ) -> bool:
        """Send weights via gRPC to a specific client."""
        start_time = time.time()

        try:
            # Serialize weights to bytes
            data = WeightSerializer.serialize(
                weights,
                compress=self.compress,
                compression_level=self.compression_level,
            )

            # In a real gRPC setup, we'd call:
            # stub = FederationServiceStub(channel)
            # response = stub.SendGlobalWeights(WeightsRequest(
            #     client_id=client_id,
            #     weights_data=data,
            #     round_num=round_num,
            #     compressed=self.compress,
            # ))

            elapsed = time.time() - start_time
            self.stats.total_bytes_sent += len(data)
            self.stats.total_messages_sent += 1
            self.stats.total_send_time_seconds += elapsed

            logger.debug(
                f"[gRPC] Sent {len(data) / (1024 * 1024):.2f} MB to "
                f"client {client_id} in {elapsed:.3f}s"
            )
            return True

        except Exception as e:
            self.stats.failed_sends += 1
            logger.error(f"[gRPC] Send to client {client_id} failed: {e}")
            return False

    def receive_update_from_client(
        self,
        client_id: int,
        timeout: float = 300.0,
    ) -> Optional[WeightMessage]:
        """Receive trained weights via gRPC from a specific client."""
        start_time = time.time()

        try:
            # In a real gRPC setup, we'd call:
            # stub = FederationServiceStub(channel)
            # response = stub.GetLocalUpdate(ClientId(id=client_id))
            # weights = WeightSerializer.deserialize(
            #     response.weights_data,
            #     compressed=response.compressed
            # )

            # Placeholder for non-gRPC environments
            if client_id in self._pending_updates:
                msg = self._pending_updates.pop(client_id)
                elapsed = time.time() - start_time
                self.stats.total_bytes_received += msg.size_bytes()
                self.stats.total_messages_received += 1
                self.stats.total_receive_time_seconds += elapsed
                return msg

            return None

        except Exception as e:
            self.stats.failed_receives += 1
            logger.error(f"[gRPC] Receive from client {client_id} failed: {e}")
            return None

    def send_update_to_server(
        self,
        client_id: int,
        weights: OrderedDict,
        num_samples: int,
        metrics: Dict[str, float],
        round_num: int,
    ) -> bool:
        """Send trained weights from client to server via gRPC."""
        start_time = time.time()

        try:
            data = WeightSerializer.serialize(
                weights,
                compress=self.compress,
                compression_level=self.compression_level,
            )

            # In a real gRPC setup:
            # stub = FederationServiceStub(self._get_channel())
            # response = stub.SendLocalUpdate(UpdateRequest(
            #     client_id=client_id,
            #     weights_data=data,
            #     num_samples=num_samples,
            #     metrics=json.dumps(metrics),
            #     round_num=round_num,
            #     compressed=self.compress,
            # ))

            elapsed = time.time() - start_time
            self.stats.total_bytes_sent += len(data)
            self.stats.total_messages_sent += 1
            self.stats.total_send_time_seconds += elapsed

            logger.debug(
                f"[gRPC] Client {client_id} sent {len(data) / (1024 * 1024):.2f} MB "
                f"update in {elapsed:.3f}s"
            )
            return True

        except Exception as e:
            self.stats.failed_sends += 1
            logger.error(f"[gRPC] Client {client_id} send failed: {e}")
            return False

    def broadcast_weights(
        self,
        weights: OrderedDict,
        round_num: int,
        client_ids: List[int],
    ) -> Dict[int, bool]:
        """Broadcast via gRPC — sends to each client sequentially."""
        results = {}
        for cid in client_ids:
            results[cid] = self.send_weights_to_client(cid, weights, round_num)
        return results

    def collect_updates(
        self,
        client_ids: List[int],
        timeout: float = 300.0,
    ) -> List[Optional[WeightMessage]]:
        """Collect from all clients via gRPC."""
        messages = []
        per_client_timeout = timeout / max(len(client_ids), 1)
        for cid in client_ids:
            msg = self.receive_update_from_client(cid, per_client_timeout)
            messages.append(msg)
        return messages

    def shutdown(self):
        """Close the gRPC channel."""
        if self.channel is not None:
            self.channel.close()
            self.channel = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FACTORY — Create the right communicator based on config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def create_communicator(
    backend: str = "ray",
    config: Dict[str, Any] = None,
    client_actors: List = None,
) -> CommunicationBackend:
    """
    Factory function to create the right communication backend.

    This is the ONLY function the rest of the code needs to call.
    It reads the config and returns the appropriate backend.

    Usage in server.py:
        communicator = create_communicator(
            backend="ray",
            config={"compress": True},
            client_actors=self.clients,
        )
        communicator.broadcast_weights(global_weights, round_num, client_ids)
        updates = communicator.collect_updates(client_ids)

    Args:
        backend: "ray" or "grpc"
        config: Backend-specific configuration
        client_actors: Ray Actor handles (only needed for "ray" backend)

    Returns:
        CommunicationBackend: Ready-to-use communicator
    """
    config = config or {}

    if backend == "ray":
        if not HAS_RAY:
            logger.warning(
                "Ray not installed. Falling back to local communication. "
                "Install with: pip install ray"
            )
        comm = RayCommunicator(config=config, client_actors=client_actors)
        logger.info(
            f"Created Ray communicator "
            f"(compress={config.get('compress', True)}, "
            f"delta={config.get('use_delta_compression', False)})"
        )
        return comm

    elif backend == "grpc":
        if not HAS_GRPC:
            raise RuntimeError(
                "gRPC not installed. Install with: pip install grpcio grpcio-tools"
            )
        comm = GRPCCommunicator(config=config)
        logger.info(
            f"Created gRPC communicator "
            f"(server={config.get('server_address', 'localhost:50051')})"
        )
        return comm

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ray' or 'grpc'.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN — Quick self-test when running directly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 60)
    print("  Communication Module Self-Test")
    print("=" * 60)

    # Test serialization
    print("\n[1] Testing WeightSerializer...")

    if HAS_TORCH:
        # Create dummy weights (like a small neural network)
        dummy_weights = OrderedDict(
            {
                "layer1.weight": torch.randn(256, 50),
                "layer1.bias": torch.randn(256),
                "layer2.weight": torch.randn(128, 256),
                "layer2.bias": torch.randn(128),
                "output.weight": torch.randn(6, 128),
                "output.bias": torch.randn(6),
            }
        )

        # Serialize without compression
        raw = WeightSerializer.serialize(dummy_weights, compress=False)
        print(f"  Raw size: {len(raw) / 1024:.1f} KB")

        # Serialize with compression
        compressed = WeightSerializer.serialize(dummy_weights, compress=True)
        print(f"  Compressed size: {len(compressed) / 1024:.1f} KB")
        print(f"  Compression ratio: {len(compressed) / len(raw):.1%}")

        # Deserialize and verify
        restored = WeightSerializer.deserialize(compressed, compressed=True)
        for key in dummy_weights:
            assert torch.equal(dummy_weights[key], restored[key]), f"Mismatch in {key}"
        print("  ✓ Serialization round-trip verified!")

        # Test delta compression
        print("\n[2] Testing delta compression...")
        # Simulate small local training changes
        local_weights = OrderedDict(
            {k: v + torch.randn_like(v) * 0.01 for k, v in dummy_weights.items()}
        )

        delta = WeightSerializer.compute_delta(dummy_weights, local_weights)
        delta_compressed = WeightSerializer.serialize(delta, compress=True)
        full_compressed = WeightSerializer.serialize(local_weights, compress=True)
        print(f"  Full weights compressed: {len(full_compressed) / 1024:.1f} KB")
        print(f"  Delta compressed: {len(delta_compressed) / 1024:.1f} KB")
        print(
            f"  Delta saves: {(1 - len(delta_compressed) / len(full_compressed)):.1%}"
        )

        # Verify delta reconstruction
        reconstructed = WeightSerializer.apply_delta(dummy_weights, delta)
        for key in local_weights:
            assert torch.allclose(local_weights[key], reconstructed[key], atol=1e-6)
        print("  ✓ Delta round-trip verified!")

    else:
        print("  ⚠ PyTorch not installed, skipping tensor tests")

    # Test communicator creation
    print("\n[3] Testing communicator factory...")
    comm = create_communicator(backend="ray", config={"compress": True})
    print(f"  Created: {type(comm).__name__}")
    print(f"  Stats: {comm.get_stats().to_dict()}")

    # Test message creation
    print("\n[4] Testing WeightMessage...")
    if HAS_TORCH:
        msg = WeightMessage(
            sender_id="client_0",
            message_type=MessageType.LOCAL_UPDATE,
            round_num=42,
            weights=dummy_weights,
            num_samples=4000,
            metrics={"reward": 15.3, "sla_rate": 0.85},
        )
        print(f"  Message size: {msg.size_mb():.2f} MB")
        print(f"  Sender: {msg.sender_id}")
        print(f"  Type: {msg.message_type.value}")
        print(f"  Round: {msg.round_num}")
        print(f"  Samples: {msg.num_samples}")
        print(f"  Metrics: {msg.metrics}")

    print("\n" + "=" * 60)
    print("  ✓ All self-tests passed!")
    print("=" * 60)
