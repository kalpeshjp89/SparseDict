import pytest
import torch

from models.losses.balance_loss import AuxiliaryLoadBalanceLoss


def test_collapsed_routing_exposes_raw_and_documented_weighted_losses():
    gates = torch.zeros(100, 4)
    gates[:, 0] = 1.0

    raw_loss = AuxiliaryLoadBalanceLoss(weight=1.0)(gates)
    documented_loss = AuxiliaryLoadBalanceLoss(weight=0.01)(gates)

    assert raw_loss.item() == pytest.approx(6.0)
    assert documented_loss.item() == pytest.approx(0.06)
    assert raw_loss.item() > 0.5
    assert documented_loss.item() < 0.5


def test_higher_weight_can_meet_weighted_threshold_without_changing_formula():
    gates = torch.zeros(100, 4)
    gates[:, 0] = 1.0

    increased_weight_loss = AuxiliaryLoadBalanceLoss(weight=0.1)(gates)

    assert increased_weight_loss.item() == pytest.approx(0.6)
    assert increased_weight_loss.item() > 0.5


def test_balanced_routing_has_zero_loss():
    gates = torch.full((12, 4), 0.25)

    assert AuxiliaryLoadBalanceLoss()(gates).item() == pytest.approx(0.0)


def test_imbalance_loss_backpropagates_through_importance():
    gates = torch.tensor(
        [[0.7, 0.3, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0]],
        requires_grad=True,
    )

    loss = AuxiliaryLoadBalanceLoss()(gates)
    loss.backward()

    assert gates.grad is not None
    assert torch.isfinite(gates.grad).all()
    assert torch.count_nonzero(gates.grad).item() > 0


@pytest.mark.parametrize(
    ("gates", "message"),
    [
        (torch.ones(4), "item and expert dimensions"),
        (torch.ones(3, 1), "at least two experts"),
        (torch.empty(0, 4), "at least one routed item"),
        (torch.tensor([[float("nan"), 0.0]]), "finite values"),
        (torch.tensor([[-1.0, 2.0]]), "non-negative"),
        (torch.zeros(2, 4), "positive total gate weight"),
    ],
)
def test_rejects_invalid_gates(gates, message):
    with pytest.raises(ValueError, match=message):
        AuxiliaryLoadBalanceLoss()(gates)


def test_rejects_negative_loss_weight():
    with pytest.raises(ValueError, match="weight must be non-negative"):
        AuxiliaryLoadBalanceLoss(weight=-0.1)
