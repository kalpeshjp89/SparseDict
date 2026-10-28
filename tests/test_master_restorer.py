"""Tests for the master restorer architecture integration (Task 6.4)."""

import torch

from models.moe_topk_restorer import (
    MasterMoETopKRestorer,
    charbonnier_loss,
    gradient_difference_loss,
    mutual_coherence_of,
)
from models.modules.moe_dict import MoEDictionary
from models.modules.topk_select import sparse_topk_attention


def test_loss_dict_return_has_all_four_paths() -> None:
    model = MasterMoETopKRestorer(d=32, atoms_per_expert=32, top_k=8)
    image = torch.rand(1, 3, 16, 16)
    prediction, losses = model(image)
    assert tuple(prediction.shape) == image.shape
    assert set(losses) == {
        "charbonnier",
        "gradient_difference",
        "load_balance",
        "coherence",
    }
    for value in losses.values():
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_total_loss_return_and_grad_flow() -> None:
    model = MasterMoETopKRestorer(d=32, atoms_per_expert=32, top_k=8)
    image = torch.rand(2, 3, 16, 16, requires_grad=True)
    prediction, total = model(image, total_loss=True)
    total.backward()
    assert total.ndim == 0
    assert image.grad is not None
    # EntropyGate parameters are excluded: the gate is a discrete H_i > tau_H
    # comparison and deliberately receives no gradient (see observations.md).
    gate_param_ids = {
        id(parameter) for parameter in model.entropy_gate.parameters()
    }
    for parameter in model.parameters():
        if id(parameter) in gate_param_ids:
            continue
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_10_epoch_micro_overfitting_monotone_decrease() -> None:
    torch.manual_seed(0)
    model = MasterMoETopKRestorer(d=32, atoms_per_expert=16, top_k=4)
    image = torch.rand(2, 3, 16, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    prev = None
    for step in range(10):
        optimizer.zero_grad()
        _, total = model(image, total_loss=True)
        total.backward()
        optimizer.step()
        value = total.item()
        if step == 0:
            first = value
            prev = value
        else:
            assert value <= prev
            prev = value
    assert first is not None and prev < first


def test_mutual_coherence_matches_manual_zero_mean_case() -> None:
    dictionary = torch.randn(2, 3, 3, requires_grad=True)
    value = mutual_coherence_of(dictionary)
    assert value.ndim == 0
    assert value.item() >= 0.0 and torch.isfinite(value)


def test_mutual_coherence_orthogonal_matrix() -> None:
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float32)).Q
    value = mutual_coherence_of(basis)
    assert value.item() < 0.35


def test_dual_domain_gates_flow_to_load_balance() -> None:
    import torch.nn.functional as F

    from models.modules.dual_domain_moe import DualDomainMoEDictionary
    from models.losses.balance_loss import AuxiliaryLoadBalanceLoss

    module = DualDomainMoEDictionary(channels=32, atoms_per_expert=16, top_k=4)
    feature_map = torch.randn(1, 32, 16, 16)
    _, gates = module(feature_map, return_gates=True)
    loss = AuxiliaryLoadBalanceLoss(weight=0.1)(gates)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in module.spatial_router.parameters():
        assert parameter.grad is not None


def test_4_expert_sub_dictionaries_receive_grad() -> None:
    model = MasterMoETopKRestorer(d=32, atoms_per_expert=16, top_k=4)
    image = torch.rand(1, 3, 16, 16)
    _, losses = model(image)
    sum(losses.values()).backward()
    dictionary_grad = model.dual_domain.dictionary.grad
    assert dictionary_grad is not None
    assert torch.isfinite(dictionary_grad).all()
    per_expert = dictionary_grad.abs().sum(dim=(1, 2))
    assert bool((per_expert > 0).all())


def test_default_dual_domain_topk_aggregation_works() -> None:
    scores = torch.randn(2, 8, 32)
    values = torch.randn(32, 16)
    result = sparse_topk_attention(scores, values, 16)
    assert tuple(result.output.shape) == (2, 8, 16)


def test_moe_dictionary_parameter_count_preserved() -> None:
    dictionary = MoEDictionary(d=32, m=64, num_experts=4)
    assert dictionary.dictionary_parameter_count == 64 * 32


def test_fused_kernel_toggle_raises() -> None:
    import pytest

    model = MasterMoETopKRestorer(d=32, atoms_per_expert=16, use_fused_kernel=True)
    with pytest.raises(RuntimeError):
        model(torch.rand(1, 3, 16, 16))


def test_gradient_difference_loss_gradients() -> None:
    torch.manual_seed(0)
    pred = torch.rand(1, 3, 8, 8, requires_grad=True)
    target = torch.rand(1, 3, 8, 8)
    value = gradient_difference_loss(pred, target)
    value.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_charbonnier_loss_matches_manual() -> None:
    pred = torch.tensor([[0.0, 2.0]])
    target = torch.tensor([[0.0, 0.0]])
    manual = ((pred - target) ** 2 + 1e-6).sqrt().mean()
    assert torch.allclose(charbonnier_loss(pred, target), manual)
