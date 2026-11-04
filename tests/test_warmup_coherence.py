import torch

from scripts.warmup_coherence import _load_dictionary, optimize_dictionary
from models.moe_topk_restorer import mutual_coherence_of


def test_load_dictionary_accepts_stage_a_and_model_checkpoints():
    dictionary = torch.randn(4, 8, 8)

    loaded = _load_dictionary({"expert_dictionary": dictionary}, torch.device("cpu"))
    assert loaded.shape == (4, 8, 8)

    loaded_from_model = _load_dictionary(
        {"model_state_dict": {"dual_domain.dictionary": dictionary}},
        torch.device("cpu"),
    )
    assert torch.equal(loaded, loaded_from_model)


def test_optimize_dictionary_stops_when_coherence_target_already_met():
    dictionary = torch.eye(8).reshape(2, 4, 8)

    result, history, target_met = optimize_dictionary(
        dictionary,
        learning_rate=1e-2,
        max_steps=5,
        target=0.1,
        check_every=1,
    )

    assert target_met
    assert history == []
    assert torch.allclose(result, dictionary)
    assert mutual_coherence_of(result).item() == 0.0
