"""Task 5.4b GPU tests: fused top-k attention forward parity and backward.

Implements the validation plan in
``05_fused_topk_autograd_design.md`` section 8. Every CUDA test uses the
standard skip-if pattern; the CPU tests run everywhere and cover the
wrapper's fail-fast validation.
"""

import pytest
import torch

from kernels.fused_topk_attn import fused_topk_attention
from models.modules.topk_select import sparse_topk_attention

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU (T4) required"
)


def _seeded_inputs(
    batch,
    heads,
    n,
    m,
    dk,
    dv,
    dtype=torch.float32,
    device="cuda",
    seed=1234,
):
    torch.manual_seed(seed)
    queries = torch.randn(batch, heads, n, dk, device=device, dtype=dtype)
    keys = torch.randn(batch, heads, m, dk, device=device, dtype=dtype)
    values = torch.randn(batch, heads, m, dv, device=device, dtype=dtype)
    return queries, keys, values


def _dense_scores(queries, keys):
    return queries @ keys.transpose(-2, -1)


def _reference_output(queries, keys, values, k, selection="hard"):
    scores = _dense_scores(queries, keys)
    return sparse_topk_attention(scores, values, k, selection=selection).output


# --- compile smoke -----------------------------------------------------------


@requires_cuda
def test_compile_smoke_fp32():
    queries, keys, values = _seeded_inputs(1, 1, 512, 256, 32, 24)
    output = fused_topk_attention(queries, keys, values, 16)
    assert output.shape == (1, 1, 512, 24)
    assert torch.isfinite(output).all()


# --- forward parity ----------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "m", "k", "dv"),
    [(1, 13, 1, 24), (7, 64, 5, 67), (65, 129, 17, 33), (3, 73, 73, 7),
     (4096, 256, 16, 64)],
)
@requires_cuda
def test_forward_parity_fp32(n, m, k, dv):
    queries, keys, values = _seeded_inputs(1, 1, n, m, 64, dv, dtype=torch.float32)
    actual = fused_topk_attention(queries, keys, values, k)
    expected = _reference_output(queries, keys, values, k)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=0.0), (
        f"max abs err {(actual - expected).abs().max().item()}"
    )


@requires_cuda
def test_forward_parity_fp64():
    queries, keys, values = _seeded_inputs(
        1, 2, 32, 24, 8, 6, dtype=torch.float64, device="cuda"
    )
    actual = fused_topk_attention(queries, keys, values, 5)
    expected = _reference_output(queries, keys, values, 5)
    assert actual.dtype == torch.float64
    assert torch.allclose(actual, expected, atol=1e-9, rtol=0.0)


# --- gradcheck (float64) -----------------------------------------------------


@pytest.mark.parametrize(
    ("n", "m", "k", "dk", "dv"),
    [(8, 12, 5, 4, 3), (16, 32, 8, 8, 8)],
)
@requires_cuda
def test_gradcheck_fp64(n, m, k, dk, dv):
    torch.manual_seed(7)
    queries = torch.randn(1, 1, n, dk, dtype=torch.float64, device="cuda",
                          requires_grad=True)
    keys = torch.randn(1, 1, m, dk, dtype=torch.float64, device="cuda",
                       requires_grad=True)
    values = torch.randn(1, 1, m, dv, dtype=torch.float64, device="cuda",
                         requires_grad=True)
    torch.autograd.gradcheck(
        fused_topk_attention,
        (queries, keys, values, k),
        eps=1e-6,
        atol=1e-4,
        nondet_tol=1e-5,
    )


# --- analytical backward vs autograd-of-reference ----------------------------


@requires_cuda
def test_backward_matches_reference_autograd_fp64():
    batch, heads, n, m, k, dk, dv = 2, 3, 32, 24, 7, 8, 6
    torch.manual_seed(42)
    queries = torch.randn(batch, heads, n, dk, dtype=torch.float64, device="cuda",
                          requires_grad=True)
    keys = torch.randn(batch, heads, m, dk, dtype=torch.float64, device="cuda",
                       requires_grad=True)
    values = torch.randn(batch, heads, m, dv, dtype=torch.float64, device="cuda",
                         requires_grad=True)
    weights = torch.randn(batch, heads, n, dv, dtype=torch.float64, device="cuda")

    fused_out = fused_topk_attention(queries, keys, values, k)
    fused_q, fused_k, fused_v = torch.autograd.grad(
        (fused_out * weights).sum(), (queries, keys, values)
    )

    ref_queries = queries.detach().clone().requires_grad_(True)
    ref_keys = keys.detach().clone().requires_grad_(True)
    ref_values = values.detach().clone().requires_grad_(True)
    ref_scores = _dense_scores(ref_queries, ref_keys)
    ref_out = sparse_topk_attention(ref_scores, ref_values, k).output
    ref_q, ref_k, ref_v = torch.autograd.grad(
        (ref_out * weights).sum(), (ref_queries, ref_keys, ref_values)
    )

    for name, actual, expected in (
        ("dQ", fused_q, ref_q),
        ("dK", fused_k, ref_k),
        ("dV", fused_v, ref_v),
    ):
        err = (actual - expected).abs().max().item()
        assert err < 1e-4, f"{name} max abs err {err}"


# --- edge cases --------------------------------------------------------------


@requires_cuda
def test_full_width_k_equals_m_fp32():
    queries, keys, values = _seeded_inputs(1, 1, 32, 32, 8, 5)
    actual = fused_topk_attention(queries, keys, values, 32)
    expected = _reference_output(queries, keys, values, 32)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=0.0)


@requires_cuda
def test_padded_candidates_zero_mass():
    # k < m forces arr_len padding lanes; output must not contain NaN paths
    queries, keys, values = _seeded_inputs(1, 1, 8, 5, 4, 4)
    output = fused_topk_attention(queries, keys, values, 2)
    assert torch.isfinite(output).all()


# --- wrapper validation (CPU-runnable) ---------------------------------------


def test_wrapper_requires_cuda_tensors():
    queries = torch.rand(1, 1, 4, 2)
    keys = torch.rand(1, 1, 5, 2)
    values = torch.rand(1, 1, 5, 3)
    with pytest.raises(ValueError, match="requires CUDA"):
        fused_topk_attention(queries, keys, values, 2)


@pytest.mark.parametrize(
    ("queries", "keys", "values", "k", "message"),
    [
        (torch.rand(1, 4, 2), torch.rand(5, 2), torch.rand(5, 3), 2, "rank-4"),
        (torch.rand(1, 2, 4, 2), torch.rand(1, 2, 5, 3), torch.rand(1, 2, 5, 3), 2,
         "feature sizes must match"),
        (torch.rand(1, 2, 4, 2), torch.rand(1, 2, 5, 2), torch.rand(1, 2, 6, 3), 2,
         "key/value atom counts must match"),
        (torch.rand(1, 2, 4, 2), torch.rand(1, 2, 5, 2), torch.rand(1, 2, 5, 3), 6,
         r"k must be an integer"),
        (torch.rand(1, 2, 4, 2), torch.rand(1, 2, 5, 2), torch.rand(1, 2, 5, 3), 2,
         "requires CUDA"),
    ],
)
def test_wrapper_rejects_invalid_geometry(queries, keys, values, k, message):
    with pytest.raises(ValueError, match=message):
        fused_topk_attention(queries, keys, values, k)
