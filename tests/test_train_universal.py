from types import SimpleNamespace

import pytest
import torch
from experiments.train_universal import (
    _apply_dictionary_schedule,
    _charbonnier_baselines,
    _effective_aux_weight,
    _make_optimizer,
    _metrics,
    _normalized_charbonnier,
    _ssim,
    _validation_acceptance,
)
from models.moe_topk_restorer import MasterMoETopKRestorer


def test_ssim_is_finite_and_one_for_identical_images():
    image = torch.rand(1, 3, 8, 10)

    score = _ssim(image, image)

    assert torch.isfinite(score)
    assert torch.allclose(score, torch.tensor(1.0), atol=1e-6)


def test_validation_metrics_report_psnr_ssim_and_l1():
    target = torch.rand(1, 3, 16, 16)
    metrics = _metrics(target, target)

    assert set(metrics) == {"psnr", "ssim", "l1"}
    assert metrics["psnr"] == 120.0
    assert metrics["ssim"] == 1.0
    assert metrics["l1"] == 0.0


def test_optimizer_disables_weight_decay_for_dictionary_only():
    model = MasterMoETopKRestorer(d=8, atoms_per_expert=8, top_k=2)
    args = SimpleNamespace(
        learning_rate=1e-3, weight_decay=0.02, dictionary_lr_scale=1.0
    )

    optimizer = _make_optimizer(model, args)
    dictionary = model.dual_domain.dictionary
    dictionary_groups = [
        group
        for group in optimizer.param_groups
        if any(parameter is dictionary for parameter in group["params"])
    ]

    assert len(dictionary_groups) == 1
    assert dictionary_groups[0]["weight_decay"] == 0.0
    assert optimizer.param_groups[0]["weight_decay"] == 0.02


def test_charbonnier_baselines_come_from_first_epoch_task_means():
    history = [
        {
            "training_by_task": {
                "denoise": {"charbonnier": 0.2},
                "sr": {"charbonnier": 0.4},
            }
        },
        {"training_by_task": {"denoise": {"charbonnier": 0.1}}},
    ]

    assert _charbonnier_baselines(history) == {"denoise": 0.2, "sr": 0.4}
    assert _charbonnier_baselines([]) == {}


def test_charbonnier_is_normalized_by_first_epoch_baseline():
    loss = torch.tensor(0.6)

    normalized = _normalized_charbonnier(
        loss, "dehaze", epoch=1, baselines={"dehaze": 0.2}, enabled=True
    )

    assert normalized.item() == 3.0
    assert _normalized_charbonnier(
        loss, "dehaze", epoch=0, baselines={}, enabled=True
    ) is loss


def test_charbonnier_normalization_fails_for_missing_task_baseline():
    with pytest.raises(ValueError, match="no initial-epoch"):
        _normalized_charbonnier(
            torch.tensor(0.5), "lowlight", epoch=1, baselines={}, enabled=True
        )


def test_auxiliary_weight_increases_only_after_expert_load_threshold():
    balanced = torch.tensor([0.45, 0.2, 0.2, 0.15])
    collapsed = torch.tensor([0.451, 0.2, 0.2, 0.149])

    assert _effective_aux_weight(balanced, 0.01, 0.05, 0.45) == 0.01
    assert _effective_aux_weight(collapsed, 0.01, 0.05, 0.45) == 0.05


def test_validation_acceptance_requires_coherence_and_monotonic_task_metrics():
    history = [
        {
            "validation": {
                "mutual_coherence_max": 0.1,
                "tasks": {"sr2": {"psnr": 20.0, "ssim": 0.7}},
            }
        },
        {
            "validation": {
                "mutual_coherence_max": 0.1,
                "tasks": {"sr2": {"psnr": 21.0, "ssim": 0.8}},
            }
        },
    ]

    assert _validation_acceptance(history, 0.15)["passed"]
    history[-1]["validation"]["tasks"]["sr2"]["psnr"] = 19.0
    result = _validation_acceptance(history, 0.15)
    assert not result["passed"]
    assert not result["per_task_validation_monotonic"]["sr2"]


def test_dictionary_schedule_freezes_then_unfreezes_with_scaled_lr():
    from types import SimpleNamespace

    model = MasterMoETopKRestorer(d=8, atoms_per_expert=8, top_k=2)
    args = SimpleNamespace(
        phase="fine_tune",
        learning_rate=1e-3,
        weight_decay=0.0,
        dictionary_lr_scale=0.1,
        freeze_dictionary_epochs=2,
    )
    optimizer = _make_optimizer(model, args)
    dictionary = model.dual_domain.dictionary

    _apply_dictionary_schedule(optimizer, model, args, epoch=0)
    assert not dictionary.requires_grad
    _apply_dictionary_schedule(optimizer, model, args, epoch=1)
    assert not dictionary.requires_grad
    _apply_dictionary_schedule(optimizer, model, args, epoch=2)
    assert dictionary.requires_grad
    group = [g for g in optimizer.param_groups if any(p is dictionary for p in g["params"])][0]
    assert group["lr"] == 1e-4


def test_acceptance_eps_allows_tiny_task_dips():
    history = [
        {
            "validation": {
                "mutual_coherence_max": 0.10,
                "tasks": {"sr2": {"psnr": 25.0, "ssim": 0.80}},
            }
        },
        {
            "validation": {
                "mutual_coherence_max": 0.10,
                "tasks": {"sr2": {"psnr": 24.995, "ssim": 0.7999}},
            }
        },
    ]
    strict = _validation_acceptance(history, 0.20)
    assert not strict["passed"]
    tolerant = _validation_acceptance(history, 0.20, eps_psnr=0.01, eps_ssim=0.001)
    assert tolerant["passed"]
    assert tolerant["monotonic_eps_psnr"] == 0.01
    assert tolerant["per_task_validation_monotonic"]["sr2"]


def test_lagrangian_coherence_weight_dormant_below_threshold_then_exponential():
    import math

    from experiments.train_universal import lagrangian_coherence_weight

    # Dormant while mu <= threshold.
    assert lagrangian_coherence_weight(0.17, 0.05, 0.18, 15.0) == pytest.approx(0.05)
    assert lagrangian_coherence_weight(0.18, 0.05, 0.18, 15.0) == pytest.approx(0.05)
    # Exponential past the threshold: lambda0 * exp(excess * gain).
    assert lagrangian_coherence_weight(0.2169, 0.05, 0.18, 15.0) == pytest.approx(
        0.05 * math.exp((0.2169 - 0.18) * 15.0)
    )
    # Monotonically increasing in mu and gain.
    assert lagrangian_coherence_weight(0.23, 0.05, 0.18, 15.0) > (
        lagrangian_coherence_weight(0.20, 0.05, 0.18, 15.0)
    )


def test_coherence_squeeze_lowers_mu_and_renormalizes_atoms():
    from experiments.train_universal import _coherence_squeeze
    from models.moe_topk_restorer import mutual_coherence_of

    torch.manual_seed(7)
    model = MasterMoETopKRestorer(d=16, atoms_per_expert=16, top_k=2)
    dictionary = model.dual_domain.dictionary
    dictionary.requires_grad_(False)
    before = float(mutual_coherence_of(dictionary.detach()).item())

    record = _coherence_squeeze(
        model, steps=10, learning_rate=0.005, max_temperature=0.05
    )

    after = float(mutual_coherence_of(dictionary.detach()).item())
    assert record["steps"] == 10
    assert record["learning_rate"] == pytest.approx(0.005)
    assert record["initial_mu"] == pytest.approx(before)
    assert after < before  # squeeze pulls mu down
    # Atom rows are unit-normalized after the squeeze.
    norms = dictionary.detach().norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # requires_grad state is restored.
    assert dictionary.requires_grad is False


def test_coherence_squeeze_restores_requires_grad_true():
    from experiments.train_universal import _coherence_squeeze

    model = MasterMoETopKRestorer(d=8, atoms_per_expert=8, top_k=2)
    model.dual_domain.dictionary.requires_grad_(True)
    _coherence_squeeze(model, steps=2, learning_rate=0.005)
    assert model.dual_domain.dictionary.requires_grad is True


def test_coherence_squeeze_rejects_invalid_arguments():
    from experiments.train_universal import _coherence_squeeze

    model = MasterMoETopKRestorer(d=8, atoms_per_expert=8, top_k=2)
    with pytest.raises(ValueError):
        _coherence_squeeze(model, steps=0, learning_rate=0.005)
    with pytest.raises(ValueError):
        _coherence_squeeze(model, steps=5, learning_rate=0.0)
