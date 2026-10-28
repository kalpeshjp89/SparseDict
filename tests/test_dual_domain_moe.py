"""Tests for the spatial-frequency dual-domain MoE dictionary (Task 6.2)."""

import pytest
import torch

from models.modules.dual_domain_moe import (
    DualDomainMoEDictionary,
    haar_dwt2,
    haar_idwt2,
)


def test_haar_roundtrip_is_inverse() -> None:
    x = torch.randn(2, 3, 8, 6, dtype=torch.float64)
    ll, lh, hl, hh = haar_dwt2(x)
    reconstructed = haar_idwt2(ll, lh, hl, hh)
    assert torch.allclose(reconstructed, x, atol=1e-12)


def test_haar_subbands_have_half_spatial_size() -> None:
    x = torch.randn(1, 4, 10, 6)
    ll, lh, hl, hh = haar_dwt2(x)
    for band in (ll, lh, hl, hh):
        assert tuple(band.shape) == (1, 4, 5, 3)


def test_haar_odd_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError):
        haar_dwt2(torch.randn(1, 1, 5, 4))
    with pytest.raises(ValueError):
        haar_dwt2(torch.randn(1, 1, 4, 5))


def test_haar_ll_is_lowpass_average() -> None:
    x = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    ll, _, _, _ = haar_dwt2(x)
    expected = x.reshape(1, 1, 2, 2, 2, 2).sum(dim=(-1, -3)) / 2.0
    assert torch.allclose(ll, expected, atol=1e-12)


def test_module_forward_shape() -> None:
    module = DualDomainMoEDictionary(channels=64, atoms_per_expert=32, top_k=8)
    feature_map = torch.randn(2, 64, 16, 12)
    output = module(feature_map)
    assert tuple(output.shape) == (2, 64, 16, 12)


def test_module_forward_is_differentiable() -> None:
    module = DualDomainMoEDictionary(channels=64, atoms_per_expert=16, top_k=4)
    feature_map = torch.randn(1, 64, 8, 8, requires_grad=True)
    output = module(feature_map)
    output.sum().backward()
    assert feature_map.grad is not None
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_module_rejects_odd_and_small_spatial_dims() -> None:
    module = DualDomainMoEDictionary(channels=64)
    with pytest.raises(ValueError):
        module(torch.randn(1, 64, 7, 8))
    with pytest.raises(ValueError):
        module(torch.randn(1, 64, 1, 8))
    with pytest.raises(ValueError):
        module(torch.randn(1, 32, 8, 8))


def test_gates_route_tokens_to_expert_subsets() -> None:
    module = DualDomainMoEDictionary(channels=64, atoms_per_expert=16, top_k=8)
    tokens = torch.randn(1, 64, 64)
    even = (torch.arange(64) % 2 == 0).view(1, 64, 1)
    odd = even.logical_not()
    gates = torch.cat((even, odd), dim=-1).float()
    routed = module._mixture(
        tokens,
        branch=0,
        bank=module.dictionary[:2],
        top_k=8,
        gates=gates,
    )
    uniform = module._mixture(
        tokens,
        branch=0,
        bank=module.dictionary[:2],
        top_k=8,
        gates=torch.ones(1, 64, 2),
    )
    assert tuple(routed.shape) == (1, 64, 64)
    assert not torch.allclose(routed, uniform)


def test_invalid_gates_shape_fails() -> None:
    module = DualDomainMoEDictionary(channels=64, atoms_per_expert=16)
    with pytest.raises(ValueError):
        module._mixture(
            torch.randn(1, 64, 64),
            branch=0,
            bank=module.dictionary[:2],
            top_k=8,
            gates=torch.ones(1, 64, 3),
        )


def test_dictionary_parameter_count() -> None:
    module = DualDomainMoEDictionary(channels=64, atoms_per_expert=32)
    assert module.dictionary_parameter_count == 4 * 32 * 64
