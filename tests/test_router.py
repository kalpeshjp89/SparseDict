import pytest
import torch

from models.modules.router import DegradationAwareRouter


@pytest.mark.parametrize(
    ("mode", "expected_shape"),
    [("image", (2, 4)), ("token", (2, 7, 4))],
)
def test_router_returns_sparse_normalized_gates(mode, expected_shape):
    router = DegradationAwareRouter(
        d=8, num_experts=4, active_experts=2, noise_std=0.0
    )
    router.eval()
    gates = router(torch.randn(2, 7, 8), mode=mode)

    assert gates.shape == expected_shape
    assert torch.allclose(gates.sum(dim=-1), torch.ones(expected_shape[:-1]))
    assert torch.all((gates == 0).sum(dim=-1) == 2)
    assert torch.all(gates >= 0)


def test_single_active_expert_has_unit_gate():
    router = DegradationAwareRouter(
        d=8, num_experts=4, active_experts=1, noise_std=0.0
    )
    gates = router(torch.randn(2, 3, 8))

    assert torch.all((gates != 0).sum(dim=-1) == 1)
    assert torch.equal(gates.sum(dim=-1), torch.ones(2, 3))


def test_training_noise_is_enabled_and_evaluation_is_deterministic():
    router = DegradationAwareRouter(d=8, num_experts=4, noise_std=1.0)
    x = torch.randn(2, 3, 8)

    router.train()
    torch.manual_seed(1)
    noisy_a = router(x)
    torch.manual_seed(2)
    noisy_b = router(x)
    assert not torch.equal(noisy_a, noisy_b)

    router.eval()
    assert torch.equal(router(x), router(x))


def test_router_backpropagates_through_selected_gate_weights():
    router = DegradationAwareRouter(
        d=8, num_experts=4, active_experts=2, noise_std=0.0
    )
    x = torch.randn(2, 5, 8)
    gates = router(x)
    expert_weights = torch.arange(1, 5, dtype=gates.dtype)
    loss = (gates * expert_weights).sum()
    loss.backward()

    assert router.fc1.weight.grad is not None
    assert torch.count_nonzero(router.fc1.weight.grad).item() > 0
    assert router.fc2.weight.grad is not None
    assert torch.count_nonzero(router.fc2.weight.grad).item() > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"active_experts": 0}, "active_experts must be between"),
        ({"active_experts": 5}, "active_experts must be between"),
        ({"noise_std": -0.1}, "noise_std must be non-negative"),
    ],
)
def test_rejects_invalid_router_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DegradationAwareRouter(**kwargs)


def test_rejects_invalid_input_and_mode():
    router = DegradationAwareRouter(d=8)
    with pytest.raises(ValueError, match="expected X"):
        router(torch.randn(2, 3, 7))
    with pytest.raises(ValueError, match="mode must"):
        router(torch.randn(2, 3, 8), mode="unknown")
