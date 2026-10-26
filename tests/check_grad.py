"""Gradient-flow smoke test for the dense dictionary attention baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from configs import hyperparams as hp
from models.modules.dict_attention import DenseDictCrossAttention


def test_trainable_parameters_receive_finite_nonzero_gradients() -> None:
    torch.manual_seed(int(hp.SEED.value))
    model = DenseDictCrossAttention()
    x = torch.randn(int(hp.B.value), int(hp.N.value), int(hp.D.value))
    target = torch.randn_like(x)

    loss = F.mse_loss(model(x), target)
    loss.backward()

    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_parameters
    for name, parameter in trainable_parameters:
        assert parameter.grad is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has non-finite gradients"
        assert torch.count_nonzero(parameter.grad).item() > 0, (
            f"{name} has only zero gradients"
        )
