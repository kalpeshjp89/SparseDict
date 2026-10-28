"""Tests for the low-coherence dictionary regularization (Task 6.1)."""

import pytest
import torch

from models.losses import MutualCoherenceLoss


def _orthogonal_square(size: int) -> torch.Tensor:
    basis = torch.linalg.qr(torch.randn(size, size, dtype=torch.float64)).Q
    return basis


def test_orthogonal_matrix_loss_below_threshold() -> None:
    loss = MutualCoherenceLoss()(_orthogonal_square(16))
    assert loss.item() < 1e-6


def test_collinear_matrix_loss_above_threshold() -> None:
    dictionary = torch.ones(64, 16, dtype=torch.float64)
    loss = MutualCoherenceLoss()(dictionary)
    assert loss.item() > 5.0


def test_loss_is_differentiable() -> None:
    dictionary = torch.randn(64, 16, dtype=torch.float64, requires_grad=True)
    loss = MutualCoherenceLoss()(dictionary)
    loss.backward()
    assert dictionary.grad is not None
    assert torch.isfinite(dictionary.grad).all()
    assert dictionary.grad.abs().sum() > 0


def test_batched_dictionary_returns_per_matrix_loss() -> None:
    dictionary = torch.randn(2, 64, 16, dtype=torch.float64)
    loss = MutualCoherenceLoss()(dictionary)
    assert tuple(loss.shape) == (2,)


def test_lambda_overrides_change_loss() -> None:
    dictionary = torch.randn(64, 16, dtype=torch.float64)
    default = MutualCoherenceLoss()(dictionary)
    zeroed = MutualCoherenceLoss(lambda_fp=0.0, lambda_hinge=1.0)(dictionary)
    assert default.item() != zeroed.item()


def test_non_finite_lambdas_are_rejected() -> None:
    with pytest.raises(ValueError):
        MutualCoherenceLoss(lambda_fp=float("nan"))
    with pytest.raises(ValueError):
        MutualCoherenceLoss(lambda_hinge=-1.0)


def test_invalid_dictionary_shapes_are_rejected() -> None:
    with pytest.raises(ValueError):
        MutualCoherenceLoss()(torch.ones(8, 16, dtype=torch.float64))
    with pytest.raises(ValueError):
        MutualCoherenceLoss()(torch.ones(4, 16, dtype=torch.float64))
    with pytest.raises(ValueError):
        MutualCoherenceLoss()(torch.ones(8, 16, dtype=torch.int64).float())


def test_welch_bound_value() -> None:
    loss = MutualCoherenceLoss()
    expected = ((64 - 16) / (16 * 63)) ** 0.5
    assert loss.welch_bound(64, 16) == pytest.approx(expected)
