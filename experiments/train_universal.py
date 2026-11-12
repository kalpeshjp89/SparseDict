"""Resumable Phase-7 universal multi-task training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.universal_loader import TASK_SCALES, UniversalRestorationDataset
from models.moe_topk_restorer import MasterMoETopKRestorer, mutual_coherence_of


class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    def _cpu_bytes(value: Any) -> torch.Tensor:
        # set_rng_state demands a CPU uint8 tensor; map_location=device in
        # torch.load can otherwise relocate the saved state to CUDA and make
        # resume fail with "RNG state must be a torch.ByteTensor".
        if isinstance(value, (list, tuple)):
            tensor = torch.tensor(value, dtype=torch.uint8)
        elif not isinstance(value, torch.Tensor):
            tensor = torch.as_tensor(value, dtype=torch.uint8)
        else:
            tensor = value
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if tensor.dtype != torch.uint8:
            tensor = tensor.to(torch.uint8)
        return tensor.contiguous()

    random.setstate(state["python"])
    torch.set_rng_state(_cpu_bytes(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [_cpu_bytes(cuda_state) for cuda_state in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def _ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel_size = min(11, prediction.size(-2), prediction.size(-1))
    if kernel_size % 2 == 0:
        kernel_size -= 1
    kernel_size = max(1, kernel_size)
    padding = kernel_size // 2
    coordinate = torch.arange(
        kernel_size, device=prediction.device, dtype=prediction.dtype
    ) - padding
    gaussian_1d = torch.exp(-0.5 * (coordinate / 1.5).square())
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    kernel = torch.outer(gaussian_1d, gaussian_1d)
    kernel = kernel.expand(prediction.size(1), 1, kernel_size, kernel_size)

    def filter_spatial(value: torch.Tensor) -> torch.Tensor:
        return F.conv2d(value, kernel, padding=padding, groups=value.size(1))

    mean_pred = filter_spatial(prediction)
    mean_target = filter_spatial(target)
    var_pred = filter_spatial(prediction.square()) - mean_pred.square()
    var_target = filter_spatial(target.square()) - mean_target.square()
    covariance = filter_spatial(prediction * target) - mean_pred * mean_target
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_pred * mean_target + c1) * (2 * covariance + c2)) / (
        (mean_pred.square() + mean_target.square() + c1)
        * (var_pred + var_target + c2)
    )
    return score.mean()


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(prediction, target).clamp_min(1e-12)
    return {
        "psnr": float((-10.0 * torch.log10(mse)).item()),
        "ssim": float(_ssim(prediction, target).item()),
        "l1": float(F.l1_loss(prediction, target).item()),
    }


def _model_config(args: argparse.Namespace) -> dict[str, int | float]:
    noise_std = (
        args.logit_jitter
        if args.phase == "warmup"
        or (args.phase == "fine_tune" and args.fine_tune_jitter_steps > 0)
        else 0.0
    )
    return {
        "d": args.dim,
        "atoms_per_expert": args.atoms_per_expert,
        "top_k": args.top_k,
        "noise_std": noise_std,
        "aux_weight": args.aux_weight,
        "entropy_kappa": args.entropy_kappa,
        "global_context_dim": args.global_context_dim,
    }


def _make_optimizer(
    model: MasterMoETopKRestorer, args: argparse.Namespace
) -> torch.optim.AdamW:
    dictionary = model.dual_domain.dictionary
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter is not dictionary
    ]
    return torch.optim.AdamW(
        [
            {"params": other_parameters, "weight_decay": args.weight_decay},
            # Strategy 1 phase-2: dictionary group may train at a reduced
            # rate (0 < dictionary_lr_scale <= 1) to protect the warmed
            # low-coherence geometry.
            {
                "params": [dictionary],
                "weight_decay": 0.0,
                "lr": args.learning_rate * max(0.0, args.dictionary_lr_scale),
            },
        ],
        lr=args.learning_rate,
    )


def _apply_dictionary_schedule(
    optimizer: torch.optim.Optimizer,
    model: MasterMoETopKRestorer,
    args: argparse.Namespace,
    epoch: int,
) -> None:
    """Strategy 1 two-phase curriculum. ``--freeze-dictionary-epochs N``
    freezes the dictionary (all gradients off) for epochs 1..N (1-based)
    of fine-tuning; later epochs train it at ``(lr * lr_scale)``."""
    if args.phase != "fine_tune":
        return
    frozen = int(args.freeze_dictionary_epochs) > epoch
    dictionary = model.dual_domain.dictionary
    target_lr = args.learning_rate * max(0.0, args.dictionary_lr_scale)
    for group in optimizer.param_groups:
        if any(parameter is dictionary for parameter in group["params"]):
            group["lr"] = target_lr
            break
    dictionary.requires_grad_(not frozen)
    label = "frozen" if frozen else f"lr={target_lr:.3e}"
    print(f"dictionary_schedule epoch={epoch + 1} {label}", flush=True)


def _charbonnier_baselines(history: list[dict[str, Any]]) -> dict[str, float]:
    if not history:
        return {}
    first_epoch = history[0].get("training_by_task", {})
    return {
        task: float(metrics["charbonnier"])
        for task, metrics in first_epoch.items()
        if float(metrics.get("charbonnier", 0.0)) > 0
    }


def _normalized_charbonnier(
    loss: torch.Tensor,
    task: str,
    epoch: int,
    baselines: dict[str, float],
    enabled: bool,
) -> torch.Tensor:
    if not enabled or epoch == 0:
        return loss
    if task not in baselines:
        raise ValueError(
            f"task {task!r} has no initial-epoch Charbonnier baseline; "
            "ensure every training task is represented in epoch 1"
        )
    return loss / max(baselines[task], 1e-8)


def _effective_aux_weight(
    expert_load: torch.Tensor,
    base_weight: float,
    collapsed_weight: float,
    threshold: float,
) -> float:
    return (
        collapsed_weight
        if float(expert_load.max().item()) > threshold
        else base_weight
    )


def _validation_acceptance(
    history: list[dict[str, Any]],
    coherence_target: float,
    eps_psnr: float = 0.0,
    eps_ssim: float = 0.0,
    monotonic_mode: str = "eps",
) -> dict[str, Any]:
    """Acceptance gates.

    ``monotonic_mode`` controls the per-task validation monotonicity check:
    - "strict": current >= previous (no tolerance)
    - "eps": current >= previous - eps (tiny dip allowed)
    - "first_vs_final": PSNR/SSIM final >= initial for every task
    - "moving_avg_3": 3-epoch moving average is non-decreasing
    - "combined": pass if first_vs_final OR moving_avg_3 holds
    """
    if not history:
        raise ValueError("cannot evaluate acceptance without completed epochs")
    curves: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for record in history:
        for task, metrics in record["validation"]["tasks"].items():
            curves[task].append((float(metrics["psnr"]), float(metrics["ssim"])))
    # strict / eps: consecutive-pair comparison
    if monotonic_mode in ("strict", "eps"):
        monotonic_tasks = {
            task: all(
                current_psnr >= previous_psnr - (eps_psnr if monotonic_mode == "eps" else 0.0)
                and current_ssim >= previous_ssim - (eps_ssim if monotonic_mode == "eps" else 0.0)
                for (previous_psnr, previous_ssim), (current_psnr, current_ssim) in zip(
                    values, values[1:]
                )
            )
            for task, values in curves.items()
        }
    elif monotonic_mode == "first_vs_final":
        monotonic_tasks = {
            task: (values[-1][0] >= values[0][0] and values[-1][1] >= values[0][1])
            for task, values in curves.items()
        }
    elif monotonic_mode == "moving_avg_3":
        monotonic_tasks = {}
        for task, values in curves.items():
            if len(values) < 3:
                monotonic_tasks[task] = True
                continue
            # 3-epoch moving average, non-decreasing
            ma = []
            for i in range(2, len(values)):
                ma.append((
                    (values[i-2][0] + values[i-1][0] + values[i][0]) / 3.0,
                    (values[i-2][1] + values[i-1][1] + values[i][1]) / 3.0,
                ))
            monotonic_tasks[task] = all(
                ma[i][0] >= ma[i-1][0] - 1e-9 and ma[i][1] >= ma[i-1][1] - 1e-9
                for i in range(1, len(ma))
            )
    elif monotonic_mode == "combined":
        fvf = {
            task: (values[-1][0] >= values[0][0] and values[-1][1] >= values[0][1])
            for task, values in curves.items()
        }
        ma3 = {}
        for task, values in curves.items():
            if len(values) < 3:
                ma3[task] = True
                continue
            ma = []
            for i in range(2, len(values)):
                ma.append((
                    (values[i-2][0] + values[i-1][0] + values[i][0]) / 3.0,
                    (values[i-2][1] + values[i-1][1] + values[i][1]) / 3.0,
                ))
            ma3[task] = all(
                ma[i][0] >= ma[i-1][0] - 1e-9 and ma[i][1] >= ma[i-1][1] - 1e-9
                for i in range(1, len(ma))
            )
        monotonic_tasks = {task: fvf[task] or ma3[task] for task in curves}
    else:
        raise ValueError(f"unknown monotonic_mode: {monotonic_mode}")
    final_mu = float(history[-1]["validation"]["mutual_coherence_max"])
    coherence_met = final_mu < coherence_target
    validation_met = bool(monotonic_tasks) and all(monotonic_tasks.values())
    return {
        "passed": coherence_met and validation_met,
        "coherence_target": coherence_target,
        "final_mutual_coherence_max": final_mu,
        "coherence_met": coherence_met,
        "monotonic_mode": monotonic_mode,
        "monotonic_eps_psnr": eps_psnr,
        "monotonic_eps_ssim": eps_ssim,
        "per_task_validation_monotonic": monotonic_tasks,
        "validation_met": validation_met,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_config(args: argparse.Namespace) -> dict[str, Any]:
    init_checkpoint = args.init_checkpoint
    return {
        "phase": args.phase,
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": _file_sha256(args.train_manifest),
        "validation_manifest": str(args.validation_manifest.resolve()),
        "validation_manifest_sha256": _file_sha256(args.validation_manifest),
        "init_checkpoint": (
            str(init_checkpoint.resolve())
            if init_checkpoint is not None
            else None
        ),
        "init_checkpoint_sha256": (
            _file_sha256(init_checkpoint)
            if init_checkpoint is not None
            else None
        ),
        "dictionary_checkpoint": (
            str(args.dictionary_checkpoint.resolve())
            if args.dictionary_checkpoint is not None
            else None
        ),
        "dictionary_checkpoint_sha256": (
            _file_sha256(args.dictionary_checkpoint)
            if args.dictionary_checkpoint is not None
            else None
        ),
        "steps_per_epoch": args.steps_per_epoch,
        "patch_size": args.patch_size,
        "validation_samples_per_task": args.validation_samples_per_task,
        "learning_rate": args.learning_rate,
        "warmup_epochs": args.warmup_epochs,
        "gdl_weight": args.gdl_weight,
        "coherence_weight": args.coherence_weight,
        "coherence_target": args.coherence_target,
        "weight_decay": args.weight_decay,
        "freeze_dictionary_epochs": args.freeze_dictionary_epochs,
        "dictionary_lr_scale": args.dictionary_lr_scale,
        "monotonic_eps_psnr": args.monotonic_eps_psnr,
        "monotonic_eps_ssim": args.monotonic_eps_ssim,
        "monotonic_mode": args.monotonic_mode,
        "normalize_task_losses": args.normalize_task_losses,
        "fine_tune_jitter_steps": args.fine_tune_jitter_steps,
        "entropy_kappa": args.entropy_kappa,
        "global_context_dim": args.global_context_dim,
        "collapsed_aux_weight": args.collapsed_aux_weight,
        "collapsed_expert_threshold": args.collapsed_expert_threshold,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "log_every_steps": args.log_every_steps,
        "seed": args.seed,
        "model_config": _model_config(args),
    }


def _load_model_state_compatible(
    model, state, context: str
) -> list[str]:
    """Load a state dict, skipping keys whose shapes do not match the model.

    Needed when the architecture grows (e.g. ``--global-context-dim`` widens
    the router input): checkpoints from before the change are still usable,
    and only the newly-shaped modules keep their fresh initialization.
    Returns the list of skipped keys.
    """
    model_state = model.state_dict()
    skipped = []
    filtered = {}
    for key, value in state.items():
        expected = model_state.get(key)
        if expected is not None and tuple(expected.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        filtered[key] = value
    if skipped:
        print(
            f"[init] skipping {len(skipped)} size-mismatched keys from "
            f"{context}: {', '.join(skipped)}",
            flush=True,
        )
    model.load_state_dict(filtered, strict=False)
    return skipped


def _load_dictionary(
    model: MasterMoETopKRestorer, checkpoint_path: Path, device: torch.device
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dictionary = checkpoint.get("expert_dictionary")
    if dictionary is None:
        dictionary = checkpoint.get("dictionary")
    if dictionary is None:
        raise ValueError(
            f"{checkpoint_path} has neither expert_dictionary nor dictionary"
        )
    dictionary = torch.as_tensor(dictionary, device=device, dtype=torch.float32)
    expected = tuple(model.dual_domain.dictionary.shape)
    if tuple(dictionary.shape) == (expected[0] * expected[1], expected[2]):
        dictionary = dictionary.reshape(expected)
    if tuple(dictionary.shape) != expected:
        raise ValueError(
            f"dictionary shape {tuple(dictionary.shape)} does not match model "
            f"shape {expected}"
        )
    with torch.no_grad():
        model.dual_domain.dictionary.copy_(dictionary)


def _save_checkpoint(
    path: Path,
    model: MasterMoETopKRestorer,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    next_batch: int,
    global_step: int,
    history: list[dict[str, Any]],
    best_psnr: float,
    training_config: dict[str, Any],
    partial_training: dict[str, Any] | None = None,
) -> None:
    _atomic_checkpoint(
        path,
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": _model_config(args),
            "training_config": training_config,
            "phase": args.phase,
            "epoch": epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "best_validation_psnr": best_psnr,
            "history": history,
            "partial_training": partial_training or {},
            "trainer_status": {
                "status": "in_progress",
                "phase": args.phase,
                "epoch": epoch,
                "next_batch": next_batch,
                "global_step": global_step,
                "updated_unix": time.time(),
            },
            "rng_state": _rng_state(),
        },
    )
    _atomic_json(
        path.parent / "trainer_status.json",
        {
            "status": "in_progress",
            "phase": args.phase,
            "epoch": epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "checkpoint": str(path),
            "updated_unix": time.time(),
        },
    )


def _validation(
    model: MasterMoETopKRestorer,
    dataset: UniversalRestorationDataset,
    device: torch.device,
    max_samples_per_task: int,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    model.dual_domain.spatial_router.active_experts = 1
    dataset.set_epoch(0)
    selected: dict[str, list[int]] = defaultdict(list)
    for index, (_, task, _) in enumerate(dataset.rows):
        if len(selected[task]) < max_samples_per_task:
            selected[task].append(index)
    per_task: dict[str, dict[str, float]] = {}
    loads: list[torch.Tensor] = []
    entropy_active = []
    entropy_means = []
    entropy_thresholds = []
    for task, indices in sorted(selected.items()):
        records: list[dict[str, float]] = []
        for index in indices:
            sample = dataset[index]
            low_quality = sample.low_quality.unsqueeze(0).to(device)
            target = sample.clean.unsqueeze(0).to(device)
            with torch.inference_mode():
                prediction, _, details = model(
                    low_quality,
                    reference=target,
                    scale_factor=sample.scale_factor,
                    return_details=True,
                )
            records.append(_metrics(prediction, target))
            loads.append(details["expert_load"].cpu())
            entropy_active.append(float(details["entropy_active_fraction"].item()))
            entropy_means.append(float(details["entropy_mean"].item()))
            threshold = details["entropy_threshold"]
            if threshold is not None:
                entropy_thresholds.append(float(threshold.item()))
        per_task[task] = {
            key: sum(record[key] for record in records) / len(records)
            for key in ("psnr", "ssim", "l1")
        }
        per_task[task]["samples"] = len(records)
    if not per_task:
        raise ValueError("validation manifest has no supported tasks")
    task_metrics = list(per_task.values())
    return {
        "tasks": per_task,
        "mean_psnr": sum(item["psnr"] for item in task_metrics) / len(task_metrics),
        "mean_ssim": sum(item["ssim"] for item in task_metrics) / len(task_metrics),
        "expert_load": (
            torch.stack(loads).mean(dim=0).tolist() if loads else [0.0] * 4
        ),
        "entropy_active_fraction": (
            sum(entropy_active) / len(entropy_active) if entropy_active else 0.0
        ),
        "entropy_mean": (
            sum(entropy_means) / len(entropy_means) if entropy_means else 0.0
        ),
        "entropy_threshold": (
            sum(entropy_thresholds) / len(entropy_thresholds)
            if entropy_thresholds
            else None
        ),
        "mutual_coherence_max": float(
            mutual_coherence_of(model.dual_domain.dictionary.detach()).item()
        ),
        "seed": seed,
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _truncate_step_log(path: Path, checkpoint_step: int) -> None:
    if not path.is_file():
        return
    kept = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if int(record["global_step"]) <= checkpoint_step:
                kept.append(record)
    _write_jsonl(path, kept)


def _write_task_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "epoch",
        "global_step",
        "phase",
        "task",
        "train_loss",
        "train_charbonnier",
        "train_normalized_charbonnier",
        "train_gradient_difference",
        "train_load_balance",
        "train_coherence",
        "validation_psnr",
        "validation_ssim",
        "validation_l1",
        "expert_load_0",
        "expert_load_1",
        "expert_load_2",
        "expert_load_3",
        "train_expert_load_0",
        "train_expert_load_1",
        "train_expert_load_2",
        "train_expert_load_3",
        "train_effective_aux_weight",
        "train_entropy_active_fraction",
        "train_entropy_mean",
        "train_entropy_threshold",
        "entropy_active_fraction",
        "entropy_mean",
        "entropy_threshold",
        "mutual_coherence_max",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for task, validation in record["validation"]["tasks"].items():
                training = record["training_by_task"].get(task, {})
                row: dict[str, Any] = {
                    "epoch": record["epoch"],
                    "global_step": record["global_step"],
                    "phase": record["phase"],
                    "task": task,
                    "validation_psnr": validation["psnr"],
                    "validation_ssim": validation["ssim"],
                    "validation_l1": validation["l1"],
                    "entropy_active_fraction": record["validation"][
                        "entropy_active_fraction"
                    ],
                    "entropy_mean": record["validation"]["entropy_mean"],
                    "entropy_threshold": record["validation"]["entropy_threshold"],
                    "mutual_coherence_max": record["validation"][
                        "mutual_coherence_max"
                    ],
                }
                for name in (
                    "loss",
                    "charbonnier",
                    "normalized_charbonnier",
                    "gradient_difference",
                    "load_balance",
                    "coherence",
                    "effective_aux_weight",
                ):
                    row[f"train_{name}"] = training.get(name, "")
                for expert, value in enumerate(record["validation"]["expert_load"]):
                    row[f"expert_load_{expert}"] = value
                for expert in range(4):
                    row[f"train_expert_load_{expert}"] = training.get(
                        f"expert_load_{expert}", ""
                    )
                for name in (
                    "entropy_active_fraction",
                    "entropy_mean",
                    "entropy_threshold",
                ):
                    row[f"train_{name}"] = training.get(name, "")
                writer.writerow(row)
    os.replace(temporary, path)


def _restore_checkpoint(
    path: Path,
    model: MasterMoETopKRestorer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    expected_phase: str,
    expected_config: dict[str, int | float],
    expected_training_config: dict[str, Any],
) -> tuple[int, int, int, list[dict[str, Any]], float, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("phase") != expected_phase:
        raise ValueError(f"{path} belongs to a different training phase")
    if checkpoint.get("model_config") != expected_config:
        raise ValueError(
            f"{path} model configuration does not match the current run: "
            f"{checkpoint.get('model_config')}"
        )
    if checkpoint.get("training_config") != expected_training_config:
        raise ValueError(
            f"{path} training configuration or manifest differs from the "
            "current run; use the original settings or a new output directory"
        )
    _load_model_state_compatible(
        model, checkpoint["model_state_dict"], context="resume checkpoint"
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _restore_rng_state(checkpoint["rng_state"])
    return (
        int(checkpoint["epoch"]),
        int(checkpoint["next_batch"]),
        int(checkpoint["global_step"]),
        list(checkpoint.get("history", [])),
        float(checkpoint.get("best_validation_psnr", -math.inf)),
        dict(checkpoint.get("partial_training", {})),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("warmup", "fine_tune"), required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--dictionary-checkpoint", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--validation-samples-per-task", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--atoms-per-expert", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logit-jitter", type=float, default=0.1)
    parser.add_argument("--fine-tune-jitter-steps", type=int, default=500)
    parser.add_argument("--aux-weight", type=float, default=0.01)
    parser.add_argument("--collapsed-aux-weight", type=float, default=0.05)
    parser.add_argument("--collapsed-expert-threshold", type=float, default=0.45)
    parser.add_argument("--gdl-weight", type=float, default=0.1)
    parser.add_argument(
        "--coherence-weight",
        type=float,
        default=1e-3,
        help="weight for the raw Task 6.1 Frame-Potential/Welch loss",
    )
    parser.add_argument("--coherence-target", type=float, default=0.15)
    parser.add_argument("--global-context-dim", type=int, default=0,
                        help="enable global context injection (0=disabled)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--freeze-dictionary-epochs",
        type=int,
        default=0,
        help="Strategy 1 phase 1: freeze the dictionary for epochs 1..N of fine-tuning (0 disables)",
    )
    parser.add_argument(
        "--dictionary-lr-scale",
        type=float,
        default=1.0,
        help="Strategy 1 phase 2: dictionary param-group LR = learning-rate * this factor",
    )
    parser.add_argument(
        "--monotonic-eps-psnr",
        type=float,
        default=0.0,
        help="Strategy 4: allowed per-task PSNR dip (dB) between consecutive validation epochs (0 = strict)",
    )
    parser.add_argument(
        "--monotonic-eps-ssim",
        type=float,
        default=0.0,
        help="Strategy 4: allowed per-task SSIM dip between consecutive validation epochs (0 = strict)",
    )
    parser.add_argument(
        "--monotonic-mode",
        choices=("strict", "eps", "first_vs_final", "moving_avg_3", "combined"),
        default="eps",
        help="monotonicity gate definition (Strategy 4)",
    )
    parser.add_argument(
        "--normalize-task-losses",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--entropy-kappa", type=float, default=0.1)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.steps_per_epoch <= 0:
        parser.error("--epochs and --steps-per-epoch must be positive")
    if args.warmup_epochs < 0 or args.validation_samples_per_task <= 0:
        parser.error("--warmup-epochs must be non-negative and validation count positive")
    if args.patch_size < 8 or args.patch_size % 4:
        parser.error("--patch-size must be at least 8 and divisible by 4")
    if args.atoms_per_expert < 2 or not 1 <= args.top_k <= 2 * args.atoms_per_expert:
        parser.error("--atoms-per-expert and --top-k are incompatible")
    if args.checkpoint_every_steps <= 0 or args.log_every_steps <= 0:
        parser.error("checkpoint and log intervals must be positive")
    if args.learning_rate <= 0 or args.num_workers < 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive; weight decay and workers non-negative")
    if args.freeze_dictionary_epochs < 0 or args.dictionary_lr_scale < 0:
        parser.error("freeze epochs and dictionary lr scale must be non-negative")
    if args.monotonic_eps_psnr < 0 or args.monotonic_eps_ssim < 0:
        parser.error("monotonic epsilons must be non-negative")
    if args.fine_tune_jitter_steps < 0 or args.aux_weight <= 0:
        parser.error("fine-tune jitter steps must be non-negative and aux weight positive")
    if args.collapsed_aux_weight < args.aux_weight:
        parser.error("collapsed aux weight must be at least aux weight")
    if not 0 < args.collapsed_expert_threshold <= 1:
        parser.error("collapsed expert threshold must be in (0, 1]")
    if not 0 < args.entropy_kappa:
        parser.error("entropy kappa must be positive")
    if not math.isfinite(args.coherence_target) or args.coherence_target <= 0:
        parser.error("coherence target must be finite and positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    train_dataset = UniversalRestorationDataset(
        args.train_manifest, patch_size=args.patch_size, seed=args.seed
    )
    validation_dataset = UniversalRestorationDataset(
        args.validation_manifest, patch_size=args.patch_size, seed=args.seed + 1
    )
    for dataset in (train_dataset, validation_dataset):
        for _, task, _ in dataset.rows:
            if TASK_SCALES[task] > 1 and args.patch_size % TASK_SCALES[task]:
                parser.error(
                    f"--patch-size must be divisible by the {task} scale "
                    f"({TASK_SCALES[task]})"
                )

    _seed_everything(args.seed)
    model = MasterMoETopKRestorer(
        d=args.dim,
        atoms_per_expert=args.atoms_per_expert,
        top_k=args.top_k,
        noise_std=_model_config(args)["noise_std"],
        aux_weight=args.aux_weight,
        entropy_kappa=args.entropy_kappa,
        global_context_dim=args.global_context_dim,
    ).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "latest.pt"
    training_config = _training_config(args)
    if not args.resume and latest_path.is_file():
        parser.error(
            f"{latest_path} already exists; use --resume or choose a new output directory"
        )
    if args.resume and latest_path.is_file():
        try:
            optimizer = _make_optimizer(model, args)
            (
                start_epoch,
                first_batch,
                global_step,
                history,
                best_psnr,
                resumed_partial,
            ) = _restore_checkpoint(
                latest_path,
                model,
                optimizer,
                device,
                args.phase,
                _model_config(args),
                training_config,
            )
            print(
                f"Resuming {args.phase} at epoch={start_epoch + 1}, "
                f"batch={first_batch}, step={global_step}",
                flush=True,
            )
        except (ValueError, KeyError, RuntimeError, TypeError) as exc:
            print(
                f"Could not resume from {latest_path} ({exc}); "
                "starting a fresh run.",
                flush=True,
            )
            if args.phase == "warmup" and args.init_checkpoint is None:
                parser.error("warmup requires --init-checkpoint from Stage A dictionary seeding")
            if args.init_checkpoint is not None:
                if not args.init_checkpoint.is_file():
                    parser.error(f"initial checkpoint does not exist: {args.init_checkpoint}")
                initialized = torch.load(
                    args.init_checkpoint, map_location=device, weights_only=False
                )
                model_state = initialized.get("model_state_dict")
                if model_state is not None:
                    _load_model_state_compatible(
                        model, model_state, context="init checkpoint"
                    )
                else:
                    _load_dictionary(model, args.init_checkpoint, device)
            if args.dictionary_checkpoint is not None:
                if not args.dictionary_checkpoint.is_file():
                    parser.error(
                        f"dictionary checkpoint does not exist: {args.dictionary_checkpoint}"
                    )
                _load_dictionary(model, args.dictionary_checkpoint, device)
            optimizer = _make_optimizer(model, args)
            start_epoch, first_batch, global_step, history, best_psnr = 0, 0, 0, [], -math.inf
            resumed_partial = {}
    else:
        if args.phase == "warmup" and args.init_checkpoint is None:
            parser.error("warmup requires --init-checkpoint from Stage A dictionary seeding")
        if args.init_checkpoint is not None:
            if not args.init_checkpoint.is_file():
                parser.error(f"initial checkpoint does not exist: {args.init_checkpoint}")
            initialized = torch.load(
                args.init_checkpoint, map_location=device, weights_only=False
            )
            model_state = initialized.get("model_state_dict")
            if model_state is not None:
                _load_model_state_compatible(
                    model, model_state, context="init checkpoint"
                )
            else:
                _load_dictionary(model, args.init_checkpoint, device)
        if args.dictionary_checkpoint is not None:
            if not args.dictionary_checkpoint.is_file():
                parser.error(
                    f"dictionary checkpoint does not exist: {args.dictionary_checkpoint}"
                )
            _load_dictionary(model, args.dictionary_checkpoint, device)
        optimizer = _make_optimizer(model, args)
        start_epoch, first_batch, global_step, history, best_psnr = 0, 0, 0, [], -math.inf
        resumed_partial = {}
    step_log = args.output_dir / "steps.jsonl"
    epoch_log = args.output_dir / "metrics.jsonl"
    task_log = args.output_dir / "metrics_by_task.csv"
    _truncate_step_log(step_log, global_step)
    _write_jsonl(epoch_log, history)
    _write_task_metrics(task_log, history)
    total_batches = min(args.steps_per_epoch, len(train_dataset))
    if total_batches <= 0:
        parser.error("training manifest is empty")
    model.train()
    started = time.monotonic()
    epoch = start_epoch
    batch_index = first_batch - 1
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    task_counts: dict[str, int] = defaultdict(int)
    try:
        for epoch in range(start_epoch, args.epochs):
            _apply_dictionary_schedule(optimizer, model, args, epoch)
            train_dataset.set_epoch(epoch)
            generator = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(train_dataset), generator=generator).tolist()
            order = order[: args.steps_per_epoch]
            sampler = FixedOrderSampler(order)
            loader = DataLoader(
                train_dataset,
                batch_size=1,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(args.seed + 10_000 + epoch),
            )
            skip_batches = first_batch if epoch == start_epoch else 0
            first_batch = 0
            batch_index = skip_batches - 1
            totals: dict[str, dict[str, float]] = defaultdict(
                lambda: defaultdict(float)
            )
            task_counts: dict[str, int] = defaultdict(int)
            if epoch == start_epoch and resumed_partial:
                for task, values in resumed_partial.get("totals", {}).items():
                    totals[task].update(
                        {name: float(value) for name, value in values.items()}
                    )
                task_counts.update(
                    {
                        task: int(count)
                        for task, count in resumed_partial.get("task_counts", {}).items()
                    }
                )
                resumed_partial = {}
            for batch_index, sample in enumerate(loader):
                if batch_index < skip_batches:
                    continue
                low_quality = sample.low_quality.to(device, non_blocking=True)
                clean = sample.clean.to(device, non_blocking=True)
                task = sample.task[0]
                scale = int(sample.scale_factor[0])
                active_experts = (
                    2
                    if args.phase == "warmup" and epoch < args.warmup_epochs
                    else 1
                )
                model.dual_domain.spatial_router.active_experts = active_experts
                model.dual_domain.spatial_router.noise_std = (
                    args.logit_jitter
                    if args.phase == "warmup"
                    or (
                        args.phase == "fine_tune"
                        and global_step < args.fine_tune_jitter_steps
                    )
                    else 0.0
                )
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction, losses, details = model(
                    low_quality,
                    reference=clean,
                    scale_factor=scale,
                    return_details=True,
                )
                loss_baselines = _charbonnier_baselines(history)
                normalized_charbonnier = _normalized_charbonnier(
                    losses["charbonnier"],
                    task,
                    epoch,
                    loss_baselines,
                    args.normalize_task_losses,
                )
                effective_aux_weight = _effective_aux_weight(
                    details["expert_load"],
                    args.aux_weight,
                    args.collapsed_aux_weight,
                    args.collapsed_expert_threshold,
                )
                weighted_load_balance = losses["load_balance"] * (
                    effective_aux_weight / args.aux_weight
                )
                total_loss = (
                    normalized_charbonnier
                    + args.gdl_weight * losses["gradient_difference"]
                    + weighted_load_balance
                    + args.coherence_weight * losses["coherence"]
                )
                if not bool(torch.isfinite(total_loss)):
                    raise FloatingPointError(
                        f"non-finite loss at epoch={epoch + 1}, "
                        f"batch={batch_index}, task={task}"
                    )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                global_step += 1
                task_counts[task] += 1
                task_totals = totals[task]
                task_totals["loss"] += float(total_loss.detach().item())
                for name, value in losses.items():
                    if name == "load_balance":
                        value = weighted_load_balance
                    task_totals[name] += float(value.detach().item())
                task_totals["normalized_charbonnier"] += float(
                    normalized_charbonnier.detach().item()
                )
                task_totals["effective_aux_weight"] += effective_aux_weight
                for expert, value in enumerate(details["expert_load"]):
                    task_totals[f"expert_load_{expert}"] += float(value.item())
                task_totals["entropy_active_fraction"] += float(
                    details["entropy_active_fraction"].item()
                )
                task_totals["entropy_mean"] += float(details["entropy_mean"].item())
                threshold = details["entropy_threshold"]
                if threshold is not None:
                    task_totals["entropy_threshold"] += float(threshold.item())
                if global_step % args.log_every_steps == 0:
                    _append_jsonl(
                        step_log,
                        {
                            "phase": args.phase,
                            "epoch": epoch + 1,
                            "batch": batch_index + 1,
                            "global_step": global_step,
                            "task": task,
                            "scale_factor": scale,
                            "active_experts": active_experts,
                            "loss": float(total_loss.detach().item()),
                            "losses": {
                                name: float(value.detach().item())
                                for name, value in losses.items()
                            },
                            "normalized_charbonnier": float(
                                normalized_charbonnier.detach().item()
                            ),
                            "effective_aux_weight": effective_aux_weight,
                            "expert_load": details["expert_load"].tolist(),
                            "entropy_active_fraction": float(
                                details["entropy_active_fraction"].item()
                            ),
                            "entropy_mean": float(details["entropy_mean"].item()),
                            "entropy_threshold": (
                                float(details["entropy_threshold"].item())
                                if details["entropy_threshold"] is not None
                                else None
                            ),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                    )
                if global_step % args.checkpoint_every_steps == 0:
                    _save_checkpoint(
                        latest_path,
                        model,
                        optimizer,
                        args,
                        epoch,
                        batch_index + 1,
                        global_step,
                        history,
                        best_psnr,
                        training_config,
                        {
                            "totals": {task: dict(values) for task, values in totals.items()},
                            "task_counts": dict(task_counts),
                        },
                    )
            train_summary = {
                task: {
                    name: value / task_counts[task]
                    for name, value in sums.items()
                }
                for task, sums in totals.items()
            }
            if args.normalize_task_losses and epoch == 0:
                charbonnier_baselines = {
                    task: values["charbonnier"]
                    for task, values in train_summary.items()
                    if values.get("charbonnier", 0.0) > 0
                }
                missing_tasks = sorted(
                    {task for _, task, _ in train_dataset.rows}
                    - charbonnier_baselines.keys()
                )
                if missing_tasks:
                    raise ValueError(
                        "initial epoch did not sample all training tasks; "
                        f"missing baselines for {missing_tasks}. Increase "
                        "--steps-per-epoch."
                    )
            else:
                charbonnier_baselines = _charbonnier_baselines(history)
            for task, values in train_summary.items():
                values.setdefault(
                    "normalized_charbonnier",
                    values["charbonnier"]
                    / max(charbonnier_baselines.get(task, values["charbonnier"]), 1e-8),
                )
            validation = _validation(
                model,
                validation_dataset,
                device,
                args.validation_samples_per_task,
                args.seed,
            )
            epoch_record = {
                "phase": args.phase,
                "epoch": epoch + 1,
                "global_step": global_step,
                "active_experts": (
                    2
                    if args.phase == "warmup" and epoch < args.warmup_epochs
                    else 1
                ),
                "training_by_task": train_summary,
                "charbonnier_baselines": charbonnier_baselines,
                "validation": validation,
                "elapsed_seconds": time.monotonic() - started,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_record)
            _write_jsonl(epoch_log, history)
            _write_task_metrics(task_log, history)
            print(
                f"phase={args.phase} epoch={epoch + 1}/{args.epochs} "
                f"step={global_step} validation_psnr={validation['mean_psnr']:.3f} "
                f"ssim={validation['mean_ssim']:.4f} "
                f"mu={validation['mutual_coherence_max']:.4f}",
                flush=True,
            )
            if validation["mean_psnr"] > best_psnr:
                best_psnr = validation["mean_psnr"]
                _save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer,
                    args,
                    epoch + 1,
                    0,
                    global_step,
                    history,
                    best_psnr,
                    training_config,
                )
            _save_checkpoint(
                latest_path,
                model,
                optimizer,
                args,
                epoch + 1,
                0,
                global_step,
                history,
                best_psnr,
                training_config,
            )
    except KeyboardInterrupt:
        _save_checkpoint(
            latest_path,
            model,
            optimizer,
            args,
            epoch,
            batch_index + 1,
            global_step,
            history,
            best_psnr,
            training_config,
            {
                "totals": {task: dict(values) for task, values in totals.items()},
                "task_counts": dict(task_counts),
            },
        )
        print(
            f"Interrupted safely; resume from {latest_path} with --resume.",
            flush=True,
        )
        return 130

    acceptance = (
        _validation_acceptance(
        history,
        args.coherence_target,
        eps_psnr=args.monotonic_eps_psnr,
        eps_ssim=args.monotonic_eps_ssim,
        monotonic_mode=args.monotonic_mode,
    )
        if args.phase == "fine_tune"
        else {"required": False}
    )
    completed_status = {
        "status": "completed",
        "phase": args.phase,
        "epochs_completed": int(history[-1]["epoch"]) if history else start_epoch,
        "global_step": global_step,
        "best_validation_psnr": best_psnr,
        "checkpoint": str(latest_path),
        "acceptance": acceptance,
        "updated_unix": time.time(),
    }
    latest_checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
    latest_checkpoint["trainer_status"] = completed_status
    _atomic_checkpoint(latest_path, latest_checkpoint)
    _atomic_json(args.output_dir / "trainer_status.json", completed_status)
    print(
        "acceptance="
        + json.dumps(completed_status["acceptance"], sort_keys=True),
        flush=True,
    )
    print(f"checkpoint={latest_path}")
    print(f"metrics={epoch_log}")
    print(f"status={args.output_dir / 'trainer_status.json'}")
    if args.phase != "fine_tune":
        return 0
    return 0 if acceptance["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
