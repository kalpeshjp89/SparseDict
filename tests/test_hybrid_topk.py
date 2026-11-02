import math

import pytest
import torch

from kernels.fused_topk_attn import fused_topk_attention
from kernels.hybrid_topk import hybrid_topk_attention
from models.modules.topk_select import sparse_topk_attention

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU (T4) required"
)


def _reference(queries, keys, values, k, temperature=1.0):
    scores = queries @ keys.transpose(-2, -1)
    scores = scores / (temperature * math.sqrt(queries.shape[-1]))
    return sparse_topk_attention(scores, values, k).output


@pytest.mark.parametrize(
    ("batch", "heads", "n", "m", "dk", "dv", "k"),
    [(1, 1, 7, 129, 8, 5, 7), (2, 2, 5, 263, 16, 33, 11)],
)
@requires_cuda
def test_hybrid_forward_matches_reference(batch, heads, n, m, dk, dv, k):
    torch.manual_seed(101)
    queries = torch.randn(batch, heads, n, dk, device="cuda")
    keys = torch.randn(batch, heads, m, dk, device="cuda")
    values = torch.randn(batch, heads, m, dv, device="cuda")

    actual = hybrid_topk_attention(queries, keys, values, k)
    expected = _reference(queries, keys, values, k)

    assert actual.shape == (batch, heads, n, dv)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=0.0)


@requires_cuda
def test_hybrid_custom_autograd_matches_reference():
    torch.manual_seed(202)
    queries = torch.randn(1, 2, 9, 8, device="cuda", requires_grad=True)
    keys = torch.randn(1, 2, 137, 8, device="cuda", requires_grad=True)
    values = torch.randn(1, 2, 137, 6, device="cuda", requires_grad=True)
    grad_output = torch.randn(1, 2, 9, 6, device="cuda")

    output = fused_topk_attention(queries, keys, values, 5)
    actual_grads = torch.autograd.grad(
        (output * grad_output).sum(), (queries, keys, values)
    )

    ref_queries = queries.detach().clone().requires_grad_(True)
    ref_keys = keys.detach().clone().requires_grad_(True)
    ref_values = values.detach().clone().requires_grad_(True)
    expected_output = _reference(ref_queries, ref_keys, ref_values, 5)
    expected_grads = torch.autograd.grad(
        (expected_output * grad_output).sum(),
        (ref_queries, ref_keys, ref_values),
    )

    assert torch.allclose(output, expected_output, atol=1e-5, rtol=0.0)
    for actual, expected in zip(actual_grads, expected_grads):
        assert torch.allclose(actual, expected, atol=1e-4, rtol=0.0)


def test_hybrid_wrapper_requires_cuda():
    queries = torch.rand(1, 1, 3, 4)
    keys = torch.rand(1, 1, 5, 4)
    values = torch.rand(1, 1, 5, 2)

    with pytest.raises(ValueError, match="share a CUDA device"):
        hybrid_topk_attention(queries, keys, values, 2)
