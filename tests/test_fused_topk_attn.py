import pytest
import torch

from kernels.fused_topk_attn import (
    tiled_affinity,
    tiled_topk,
    tiled_topk_aggregate,
)


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


@pytest.mark.parametrize(
    ("shape", "k"),
    [
        ((1, 1), 1),
        ((7, 64), 5),
        ((65, 129), 17),
        ((8, 256), 64),
        ((3, 73), 73),
    ],
)
def test_tiled_topk_rejects_non_cuda_scores(shape, k):
    with pytest.raises(ValueError, match="requires CUDA"):
        tiled_topk(torch.rand(shape), k)


@pytest.mark.parametrize(
    ("scores", "k", "message"),
    [
        (torch.rand(2, 3, 4), 2, "rank-2"),
        (torch.empty(0, 4), 2, "dimensions must be non-empty"),
        (torch.rand(2, 4), 0, r"k must be an integer"),
        (torch.rand(2, 4), 5, r"k must be an integer"),
    ],
)
def test_tiled_topk_rejects_invalid_inputs(scores, k, message):
    with pytest.raises(ValueError, match=message):
        tiled_topk(scores, k)


def test_tiled_topk_aggregate_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        tiled_topk_aggregate(
            torch.rand(2, 3),
            torch.zeros(2, 4, dtype=torch.int64),
            torch.rand(5, 6),
        )


def test_tiled_topk_aggregate_rejects_non_matrix_inputs():
    with pytest.raises(ValueError, match="rank-2"):
        tiled_topk_aggregate(
            torch.rand(2, 3, 4),
            torch.zeros(2, 3, 4, dtype=torch.int64),
            torch.rand(5, 6),
        )


def test_tiled_topk_aggregate_requires_cuda():
    with pytest.raises(ValueError, match="requires CUDA"):
        tiled_topk_aggregate(
            torch.rand(2, 3),
            torch.zeros(2, 3, dtype=torch.int64),
            torch.rand(5, 6),
        )


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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires an NVIDIA CUDA runtime for Triton compilation",
)
@pytest.mark.parametrize(
    ("row_count", "key_count", "k"),
    [
        (1, 1, 1),
        (7, 64, 5),
        (65, 129, 17),
        (8, 256, 64),
        (3, 73, 73),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_tiled_topk_matches_torch_topk_indices(row_count, key_count, k, dtype):
    pytest.importorskip("triton")
    from kernels.fused_topk_attn import tiled_topk

    generator = torch.Generator(device="cuda").manual_seed(
        row_count * 10000 + key_count * 100 + k
    )
    random_values = torch.rand(
        (row_count, key_count), device="cuda", generator=generator
    )
    scores = random_values.argsort(dim=1).argsort(dim=1).to(dtype)

    actual_scores, actual_indices = tiled_topk(scores, k)
    expected_scores, expected_indices = torch.topk(scores, k, dim=1)

    assert actual_scores.shape == (row_count, k)
    assert actual_indices.shape == (row_count, k)
    assert torch.equal(actual_indices, expected_indices)
    assert torch.equal(actual_scores, expected_scores)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires an NVIDIA CUDA runtime for Triton compilation",
)
@pytest.mark.parametrize(
    ("query_count", "key_count", "k", "value_dim"),
    [
        (1, 13, 1, 24),
        (7, 64, 5, 67),
        (65, 129, 17, 33),
        (3, 73, 73, 7),
    ],
)
@pytest.mark.parametrize(
    ("score_dtype", "value_dtype"),
    [(torch.float16, torch.float16), (torch.float32, torch.float32)],
)
def test_tiled_topk_aggregate_matches_torch_reference(
    query_count, key_count, k, value_dim, score_dtype, value_dtype
):
    pytest.importorskip("triton")
    from kernels.fused_topk_attn import tiled_topk, tiled_topk_aggregate

    generator = torch.Generator(device="cuda").manual_seed(
        query_count * 10000 + key_count * 100 + k + value_dim
    )
    scores = torch.randn(
        (query_count, key_count),
        device="cuda",
        dtype=score_dtype,
        generator=generator,
    )
    values = torch.randn(
        (key_count, value_dim),
        device="cuda",
        dtype=value_dtype,
        generator=generator,
    )
    selected_scores, selected_indices = tiled_topk(scores, k)

    actual = tiled_topk_aggregate(selected_scores, selected_indices, values)
    weights = torch.softmax(selected_scores.float(), dim=1)
    selected_values = values.float()[selected_indices]
    expected = torch.sum(weights.unsqueeze(-1) * selected_values, dim=1)

    assert actual.shape == (query_count, value_dim)
    assert actual.dtype == torch.float32
    assert (actual - expected).abs().max().item() <= 1e-5
