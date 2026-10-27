import pytest
import torch

from kernels.fused_topk_attn import tiled_affinity


def test_tiled_affinity_rejects_non_matrix_inputs():
    with pytest.raises(ValueError, match="rank-2"):
        tiled_affinity(torch.rand(2, 3, 4), torch.rand(5, 4))


def test_tiled_affinity_rejects_mismatched_features():
    with pytest.raises(ValueError, match="feature sizes must match"):
        tiled_affinity(torch.rand(2, 3), torch.rand(5, 4))


def test_tiled_affinity_requires_contiguous_feature_dimension():
    queries = torch.rand(4, 2).T
    with pytest.raises(ValueError, match="feature dimensions must be contiguous"):
        tiled_affinity(queries, torch.rand(5, 4))


def test_tiled_affinity_requires_cuda():
    with pytest.raises(ValueError, match="requires CUDA"):
        tiled_affinity(torch.rand(2, 4), torch.rand(5, 4))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires an NVIDIA CUDA runtime for Triton compilation",
)
def test_tiled_affinity_matches_torch_on_cuda():
    pytest.importorskip("triton")
    from kernels.fused_topk_attn import tiled_affinity

    queries = torch.randn(67, 24, device="cuda", dtype=torch.float16)
    keys = torch.randn(129, 24, device="cuda", dtype=torch.float16)

    actual = tiled_affinity(queries, keys)
    expected = queries.float() @ keys.float().T

    assert actual.shape == (67, 129)
    assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)
