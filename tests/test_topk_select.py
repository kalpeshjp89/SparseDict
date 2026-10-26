import pytest
import torch
import torch.nn.functional as F

from models.modules.topk_select import (
    gumbel_perturbed_topk,
    hard_topk_mask,
    sparse_topk_attention,
    straight_through_topk,
)


def test_hard_topk_mask_keeps_exactly_k_scores_per_query():
    scores = torch.tensor([[1.0, 4.0, 2.0, 3.0], [8.0, 5.0, 7.0, 6.0]])

    masked_scores, indices = hard_topk_mask(scores, k=2)

    assert torch.equal(indices, torch.tensor([[1, 3], [0, 2]]))
    assert torch.equal(torch.isfinite(masked_scores).sum(dim=-1), torch.tensor([2, 2]))
    assert torch.isneginf(masked_scores).sum(dim=-1).tolist() == [2, 2]


@pytest.mark.parametrize(
    ("scores", "values"),
    [
        (torch.randn(2, 3, 7), torch.randn(7, 5)),
        (torch.randn(2, 3, 7), torch.randn(2, 7, 5)),
        (torch.randn(2, 4, 3, 7), torch.randn(2, 4, 7, 5)),
    ],
)
def test_sparse_attention_matches_selected_dense_reference(scores, values):
    result = sparse_topk_attention(scores, values, k=3)
    expected_weights = F.softmax(result.masked_scores, dim=-1)

    assert result.output.shape == (*scores.shape[:-1], values.shape[-1])
    assert result.weights.shape == scores.shape
    assert result.indices.shape == (*scores.shape[:-1], 3)
    assert torch.equal(torch.count_nonzero(result.weights, dim=-1), torch.full(
        scores.shape[:-1], 3, dtype=torch.long
    ))
    assert torch.allclose(result.weights, expected_weights)

    if values.ndim == 2:
        expected_output = expected_weights @ values
    else:
        expected_output = expected_weights @ values
    assert torch.allclose(result.output, expected_output, atol=1e-6)


def test_sparse_attention_backpropagates_through_scores_and_selected_values():
    scores = torch.randn(2, 3, 6, requires_grad=True)
    values = torch.randn(6, 4, requires_grad=True)

    result = sparse_topk_attention(scores, values, k=2)
    result.output.square().sum().backward()

    assert scores.grad is not None
    assert values.grad is not None
    assert torch.count_nonzero(scores.grad).item() > 0
    assert torch.count_nonzero(values.grad).item() > 0


@pytest.mark.parametrize(
    ("scores", "k", "message"),
    [
        (torch.randn(5), 1, "query and atom dimensions"),
        (torch.ones(2, 3, dtype=torch.int64), 1, "floating-point dtype"),
        (torch.randn(2, 3), 0, "k must be between"),
        (torch.randn(2, 3), 4, "k must be between"),
    ],
)
def test_hard_topk_rejects_invalid_scores_or_k(scores, k, message):
    with pytest.raises(ValueError, match=message):
        hard_topk_mask(scores, k)


def test_sparse_attention_rejects_invalid_value_shapes():
    scores = torch.randn(2, 3, 7)

    with pytest.raises(ValueError, match="value atom count"):
        sparse_topk_attention(scores, torch.randn(6, 4), k=2)
    with pytest.raises(ValueError, match="values must be shared"):
        sparse_topk_attention(scores, torch.randn(3, 7, 4), k=2)


def test_ste_topk_passes_gradcheck_for_unique_scores():
    scores = torch.tensor(
        [[0.13, 0.92, -0.28, 0.41], [0.84, -0.37, 0.22, 0.61]],
        dtype=torch.double,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(
        lambda values: straight_through_topk(values, k=2)[0],
        (scores,),
    )


def test_ste_only_routes_gradients_to_selected_scores():
    scores = torch.tensor(
        [[0.1, 0.9, 0.2, 0.8]], dtype=torch.double, requires_grad=True
    )
    selected_scores, indices = straight_through_topk(scores, k=2)
    selected_scores.square().sum().backward()

    expected_grad = torch.tensor([[0.0, 1.8, 0.0, 1.6]], dtype=torch.double)
    assert torch.equal(indices, torch.tensor([[1, 3]]))
    assert torch.equal(scores.grad, expected_grad)


def test_ste_gradients_reach_query_projection_and_dictionary():
    torch.manual_seed(23)
    query_projection = torch.nn.Linear(4, 3, bias=False)
    dictionary = torch.nn.Parameter(torch.randn(6, 3))
    queries = torch.randn(2, 5, 4)

    scores = query_projection(queries) @ dictionary.T
    selected_scores, _ = straight_through_topk(scores, k=2)
    selected_scores.square().sum().backward()

    assert query_projection.weight.grad is not None
    assert torch.count_nonzero(query_projection.weight.grad).item() > 0
    assert dictionary.grad is not None
    assert torch.count_nonzero(dictionary.grad).item() > 0


def test_sparse_attention_supports_ste_selection():
    scores = torch.randn(2, 3, 6, requires_grad=True)
    values = torch.randn(6, 4)

    result = sparse_topk_attention(scores, values, k=2, selection="ste")
    result.output.square().sum().backward()

    assert torch.equal(
        torch.count_nonzero(result.weights, dim=-1),
        torch.full((2, 3), 2, dtype=torch.long),
    )
    assert scores.grad is not None
    assert torch.count_nonzero(scores.grad).item() > 0


def test_sparse_attention_rejects_unknown_selection_mode():
    with pytest.raises(ValueError, match="selection must be"):
        sparse_topk_attention(torch.randn(2, 3, 5), torch.randn(5, 4), k=2, selection="other")


def test_gumbel_fallback_returns_two_topk_monte_carlo_samples():
    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4]], requires_grad=True)
    generator = torch.Generator().manual_seed(9)

    sampled_scores, sampled_indices = gumbel_perturbed_topk(
        scores, k=2, sigma=0.5, num_samples=2, generator=generator
    )

    assert sampled_scores.shape == (2, 1, 2)
    assert sampled_indices.shape == (2, 1, 2)
    assert torch.all((sampled_indices >= 0) & (sampled_indices < 4))
    sampled_scores.sum().backward()
    assert scores.grad is not None
    assert torch.count_nonzero(scores.grad).item() > 0


def test_gumbel_fallback_zero_noise_matches_deterministic_topk():
    scores = torch.randn(2, 3, 7)

    sampled_scores, sampled_indices = gumbel_perturbed_topk(
        scores, k=3, sigma=0.0, num_samples=2
    )

    expected_scores, expected_indices = torch.topk(scores, 3, dim=-1)
    assert torch.equal(sampled_indices[0], expected_indices)
    assert torch.equal(sampled_indices[1], expected_indices)
    assert torch.equal(sampled_scores[0], expected_scores)
    assert torch.equal(sampled_scores[1], expected_scores)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sigma": -0.1}, "sigma must be finite and non-negative"),
        ({"sigma": float("inf")}, "sigma must be finite and non-negative"),
        ({"num_samples": 0}, "num_samples must be positive"),
    ],
)
def test_gumbel_fallback_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        gumbel_perturbed_topk(torch.randn(2, 5), k=2, **kwargs)
