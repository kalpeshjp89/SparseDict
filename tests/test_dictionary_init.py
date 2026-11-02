import torch

import scripts.init_dictionaries as dictionary_init
from scripts.init_dictionaries import (
    _cluster_kmeans_plus_plus,
    _maximum_coherence,
)


def test_kmeans_plus_plus_returns_unit_normalized_overcomplete_dictionary():
    embeddings = torch.nn.functional.normalize(torch.randn(64, 8), dim=-1)
    dictionary = _cluster_kmeans_plus_plus(
        embeddings,
        cluster_count=16,
        device=torch.device("cpu"),
        seed=7,
        init_sample_count=32,
        iterations=2,
        chunk_size=8,
    )

    assert dictionary.shape == (16, 8)
    assert torch.allclose(
        dictionary.norm(dim=-1),
        torch.ones(16),
        atol=1e-5,
    )
    assert 0.0 <= _maximum_coherence(dictionary) <= 1.0


def test_kmeans_plus_plus_resume_continues_from_saved_center_checkpoint(
    tmp_path, monkeypatch
):
    embeddings = torch.nn.functional.normalize(torch.randn(32, 8), dim=-1)
    checkpoint_path = tmp_path / "kmeans_latest.pt"
    original_save = dictionary_init._atomic_save

    def interrupt_after_checkpoint(path, payload):
        original_save(path, payload)
        if payload.get("phase") == "initializing" and payload.get("next_center", 0) >= 8:
            raise RuntimeError("simulated runtime interruption")

    monkeypatch.setattr(dictionary_init, "_atomic_save", interrupt_after_checkpoint)
    try:
        _cluster_kmeans_plus_plus(
            embeddings,
            cluster_count=16,
            device=torch.device("cpu"),
            seed=9,
            init_sample_count=24,
            iterations=2,
            chunk_size=8,
            checkpoint_path=checkpoint_path,
            checkpoint_every_centers=4,
        )
    except RuntimeError as error:
        assert str(error) == "simulated runtime interruption"
    else:
        raise AssertionError("expected simulated interruption")

    monkeypatch.setattr(dictionary_init, "_atomic_save", original_save)
    resumed = _cluster_kmeans_plus_plus(
        embeddings,
        cluster_count=16,
        device=torch.device("cpu"),
        seed=9,
        init_sample_count=24,
        iterations=2,
        chunk_size=8,
        checkpoint_path=checkpoint_path,
        checkpoint_every_centers=4,
    )
    expected = _cluster_kmeans_plus_plus(
        embeddings,
        cluster_count=16,
        device=torch.device("cpu"),
        seed=9,
        init_sample_count=24,
        iterations=2,
        chunk_size=8,
    )
    assert torch.allclose(resumed, expected)
