"""Tests for the adaptive entropy-gated dynamic top-k router (Task 6.3)."""

import pytest
import torch

from models.modules.entropy_gate import EntropyGate


def test_constant_tensor_triggers_zero_activation() -> None:
    gate = EntropyGate(channels=64)
    gate.eval()
    features = torch.ones(1, 32, 64)
    result = gate(features)
    assert int(result.gate.sum()) == 0


def test_low_spread_noise_triggers_full_activation() -> None:
    gate = EntropyGate(channels=64)
    gate.eval()
    torch.manual_seed(0)
    # Emulate a gate pre-adapted to clean-data entropy statistics.
    with torch.no_grad():
        gate.entropy_mean.copy_(torch.tensor(4.0))
        gate.entropy_std.copy_(torch.tensor(0.05))
        gate.entropy_stats_initialized.fill_(True)
    features = torch.randn(2, 64, 64) * 0.01
    result = gate(features)
    assert result.gate.numel() == 128
    assert int(result.gate.sum()) >= 127


def test_gate_zeros_and_ones_are_valid() -> None:
    gate = EntropyGate(channels=64)
    gate.eval()
    constant = torch.ones(1, 16, 64)
    noise = torch.randn(1, 16, 64)
    assert int(gate(constant).gate.abs().sum()) == 0
    assert bool(gate(noise).gate.gt(0).any())


def test_entropy_ranges_between_zero_and_log_vocab() -> None:
    gate = EntropyGate(channels=16, vocab_size=64)
    gate.eval()
    logits = torch.randn(3, 8, 16)
    entropy = gate.shannon_entropy(logits)
    assert (entropy >= -1e-6).all()
    assert (entropy <= gate.max_entropy + 1e-6).all()


def test_ema_statistics_update_over_batches() -> None:
    gate = EntropyGate(channels=16, vocab_size=64)
    assert not bool(gate.entropy_stats_initialized)
    gate.train()
    result = gate(torch.randn(1, 8, 16))
    assert bool(gate.entropy_stats_initialized)
    assert result.threshold is not None


def test_adaptive_k_within_bounds() -> None:
    gate = EntropyGate(channels=16, vocab_size=64, k_min=8, k_max=32)
    gate.eval()
    result = gate(torch.randn(1, 24, 16))
    assert all(8 <= value <= 32 for value in result.k.flatten().tolist())


def test_uniform_logits_max_entropy() -> None:
    gate = EntropyGate(channels=16, vocab_size=64)
    gate.eval()
    logits = gate.vocab_proj(torch.zeros(1, 4, 16))
    entropy = gate.shannon_entropy(logits)
    assert torch.allclose(entropy, torch.full_like(entropy, gate.max_entropy))


def test_bypass_output_shape_and_eye_skip() -> None:
    gate = EntropyGate(channels=8, vocab_size=64)
    gate.eval()
    with torch.no_grad():
        gate.skip_proj.weight.copy_(torch.eye(8))
    result = gate(torch.ones(1, 4, 8))
    assert tuple(result.output.shape) == (1, 4, 8)


def test_forward_is_differentiable() -> None:
    gate = EntropyGate(channels=16, vocab_size=64)
    features = torch.randn(1, 8, 16, requires_grad=True)
    result = gate(features)
    result.output.sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_invalid_settings_rejected() -> None:
    with pytest.raises(ValueError):
        EntropyGate(channels=0)
    with pytest.raises(ValueError):
        EntropyGate(vocab_size=1)
    with pytest.raises(ValueError):
        EntropyGate(kappa=0.0)
    with pytest.raises(ValueError):
        EntropyGate(k_min=0)
    with pytest.raises(ValueError):
        EntropyGate(k_min=16, k_max=8)


def test_malformed_inputs_rejected() -> None:
    gate = EntropyGate(channels=16, vocab_size=64)
    gate.eval()
    with pytest.raises(ValueError):
        gate(torch.randn(16, 8))
    with pytest.raises(ValueError):
        gate(torch.randn(1, 0, 16))
    with pytest.raises(ValueError):
        gate(torch.randn(1, 8, 32))
