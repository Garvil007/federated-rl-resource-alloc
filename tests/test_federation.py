import torch
import pytest
from collections import OrderedDict
from src.federation.strategies import fed_avg, compute_weight_divergence


@pytest.fixture
def mock_weights():
    """Create mock model weights for testing."""
    return OrderedDict(
        {
            "layer1.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "layer1.bias": torch.tensor([0.5, 0.5]),
        }
    )


def test_fed_avg_equal_weights(mock_weights):
    """FedAvg with equal samples should be simple average."""
    client1 = OrderedDict(
        {
            "layer1.weight": torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
            "layer1.bias": torch.tensor([1.0, 1.0]),
        }
    )
    client2 = OrderedDict(
        {
            "layer1.weight": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            "layer1.bias": torch.tensor([0.0, 0.0]),
        }
    )
    result = fed_avg(mock_weights, [(client1, 100), (client2, 100)])
    expected_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    torch.testing.assert_close(result["layer1.weight"], expected_weight)


def test_fed_avg_weighted_by_samples(mock_weights):
    """FedAvg should weight by number of samples."""
    client1 = OrderedDict(
        {
            "layer1.weight": torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
            "layer1.bias": torch.tensor([10.0, 10.0]),
        }
    )
    client2 = OrderedDict(
        {
            "layer1.weight": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            "layer1.bias": torch.tensor([0.0, 0.0]),
        }
    )
    # client1 has 3x more samples, so result should be closer to client1
    result = fed_avg(mock_weights, [(client1, 300), (client2, 100)])
    expected_weight = torch.tensor([[7.5, 7.5], [7.5, 7.5]])
    torch.testing.assert_close(result["layer1.weight"], expected_weight)


def test_weight_divergence_zero_for_same():
    w = OrderedDict({"a": torch.tensor([1.0, 2.0])})
    assert compute_weight_divergence(w, w) == pytest.approx(0.0)


def test_weight_divergence_nonzero():
    w1 = OrderedDict({"a": torch.tensor([1.0, 0.0])})
    w2 = OrderedDict({"a": torch.tensor([0.0, 0.0])})
    assert compute_weight_divergence(w1, w2) == pytest.approx(1.0)
