import torch
import copy
from typing import List, Tuple
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
    mu: float = 0.01,  # proximal term coefficient
) -> OrderedDict:
    """
    FedProx: FedAvg + proximal term to limit client drift.

    The proximal term is applied during LOCAL training:
      loss += (mu/2) * ||w_local - w_global||^2

    Aggregation step itself is identical to FedAvg.
    """
    return fed_avg(global_weights, client_weights)


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
