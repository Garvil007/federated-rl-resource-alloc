import torch
import copy
from typing import List, Tuple, Optional
from collections import OrderedDict


def fed_avg(
    global_weights: OrderedDict,
    client_weights: List[Tuple[OrderedDict, int]],  # (weights, num_samples)
) -> OrderedDict:
    """Federated Averaging: weighted mean of client model params."""
    total_samples = sum(n for _, n in client_weights)
    new_weights = copy.deepcopy(global_weights)

    for key in new_weights:
        new_weights[key] = torch.zeros_like(new_weights[key])
        for client_w, n_samples in client_weights:
            weight = n_samples / total_samples
            new_weights[key] += client_w[key] * weight
    return new_weights


def fed_prox(
    global_weights: OrderedDict,
    client_weights: List[Tuple[OrderedDict, int]],
    mu: float = 0.01,
) -> OrderedDict:
    """FedProx: aggregation is identical to FedAvg.
    The proximal term is applied during LOCAL training in client.py.
    """
    return fed_avg(global_weights, client_weights)


def fed_nova(
    global_weights: OrderedDict,
    client_deltas: List[Tuple[OrderedDict, int, int]],
    # (weight_delta, num_samples, local_steps)
) -> OrderedDict:
    """FedNova: Normalized Averaging for heterogeneous local training.

    Unlike FedAvg which averages raw weights, FedNova normalizes each
    client's update by its number of local SGD steps. This prevents clients
    that train longer from dominating the global model.

    Reference: Wang et al., "Tackling the Objective Inconsistency Problem
    in Heterogeneous Federated Optimization" (NeurIPS 2020)
    """
    total_samples = sum(n for _, n, _ in client_deltas)
    new_weights = copy.deepcopy(global_weights)

    # Compute effective number of steps (τ_eff)
    # Each client contributes proportional to samples but normalized by steps
    tau_eff = 0.0
    for _, n_samples, local_steps in client_deltas:
        p_i = n_samples / total_samples
        tau_eff += p_i * local_steps

    for key in new_weights:
        normalized_delta = torch.zeros_like(new_weights[key])
        for delta, n_samples, local_steps in client_deltas:
            p_i = n_samples / total_samples
            # Normalize this client's delta by its local steps
            normalized_delta += p_i * delta[key] / max(local_steps, 1)

        # Scale by effective steps
        new_weights[key] = global_weights[key] + tau_eff * normalized_delta

    return new_weights


def scaffold_aggregate(
    global_weights: OrderedDict,
    client_weights: List[Tuple[OrderedDict, int]],
    client_controls: List[Optional[OrderedDict]],
    global_control: OrderedDict,
    num_clients: int,
) -> Tuple[OrderedDict, OrderedDict]:
    """Scaffold: Server-side aggregation with control variate update.

    Scaffold maintains control variates that track the difference between
    each client's gradient and the global gradient. This directly corrects
    for client drift, the main problem in heterogeneous federated learning.

    Reference: Karimireddy et al., "SCAFFOLD: Stochastic Controlled
    Averaging for Federated Learning" (ICML 2020)

    Returns:
        (new_global_weights, new_global_control)
    """
    # Step 1: Average client weights (same as FedAvg)
    new_weights = fed_avg(global_weights, client_weights)

    # Step 2: Update global control variate
    # c_global_new = c_global + (1/N) * Σ(c_i_new - c_i_old)
    new_global_control = copy.deepcopy(global_control)

    num_reporting = sum(1 for c in client_controls if c is not None)
    if num_reporting > 0:
        for key in global_control:
            control_delta = torch.zeros_like(global_control[key])
            for client_c in client_controls:
                if client_c is not None and key in client_c:
                    control_delta += client_c[key]
            # Average the new control variates
            new_global_control[key] = control_delta / num_reporting

    return new_weights, new_global_control


def compute_weight_divergence(
    global_weights: OrderedDict,
    client_weights: OrderedDict,
) -> float:
    """Measure L2 distance between global and client weights."""
    total_diff = 0.0
    for key in global_weights:
        diff = (global_weights[key] - client_weights[key]).float()
        total_diff += torch.norm(diff).item() ** 2
    return total_diff**0.5
