"""Optimize a seeded dictionary against the Welch-hinged coherence loss."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.losses import MutualCoherenceLoss
from models.moe_topk_restorer import mutual_coherence_of


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_dictionary(
    checkpoint: dict[str, Any], device: torch.device
) -> torch.Tensor:
    dictionary = checkpoint.get("expert_dictionary")
    if dictionary is None:
        dictionary = checkpoint.get("dictionary")
    if dictionary is None:
        model_state = checkpoint.get("model_state_dict", {})
        dictionary = model_state.get("dual_domain.dictionary")
    if dictionary is None:
        raise ValueError(
            "checkpoint must contain expert_dictionary, dictionary, or "
            "model_state_dict['dual_domain.dictionary']"
        )
    dictionary = torch.as_tensor(dictionary, device=device, dtype=torch.float32)
    if dictionary.ndim == 2:
        if dictionary.size(0) % 4:
            raise ValueError(
                "a flat checkpoint dictionary must divide evenly into four banks"
            )
        dictionary = dictionary.reshape(4, dictionary.size(0) // 4, dictionary.size(1))
    if dictionary.ndim < 2 or dictionary.size(-1) < 1:
        raise ValueError(f"invalid dictionary shape: {tuple(dictionary.shape)}")
    flat = dictionary.flatten(end_dim=-2)
    if flat.size(0) < flat.size(1):
        raise ValueError(
            f"dictionary must be overcomplete or square, got {tuple(flat.shape)}"
        )
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("dictionary contains non-finite values")
    return dictionary


def optimize_dictionary(
    initial_dictionary: torch.Tensor,
    *,
    learning_rate: float,
    max_steps: int,
    target: float,
    check_every: int,
    max_weight: float = 0.0,
    max_temperature: float = 0.05,
) -> tuple[torch.Tensor, list[dict[str, float | int]], bool]:
    dictionary = torch.nn.Parameter(initial_dictionary.detach().clone())
    optimizer = torch.optim.Adam([dictionary], lr=learning_rate, weight_decay=0.0)
    coherence_loss = MutualCoherenceLoss(
        lambda_max=max_weight, max_temperature=max_temperature
    )
    history: list[dict[str, float | int]] = []
    current_coherence = float(mutual_coherence_of(dictionary.detach()).item())
    if current_coherence < target:
        return dictionary.detach(), history, True

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = coherence_loss(dictionary).sum()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite coherence loss at step {step}")
        loss.backward()
        if dictionary.grad is None or not bool(torch.isfinite(dictionary.grad).all()):
            raise FloatingPointError(f"invalid dictionary gradient at step {step}")
        optimizer.step()
        with torch.no_grad():
            dictionary.copy_(F.normalize(dictionary, dim=-1))

        if step % check_every == 0 or step == max_steps:
            current_coherence = float(
                mutual_coherence_of(dictionary.detach()).item()
            )
            history.append(
                {"step": step, "loss": float(loss.detach().item()), "mu": current_coherence}
            )
            if current_coherence < target:
                return dictionary.detach(), history, True
    return dictionary.detach(), history, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="default: <checkpoint-dir>/pretrained_dictionaries_incoherent.pth",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--target", type=float, default=0.20)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument(
        "--coherence-max-weight",
        type=float,
        default=64.0,
        help="weight of the smooth-max limb; 0 reproduces the legacy loss",
    )
    parser.add_argument("--max-temperature", type=float, default=0.02)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)
    if not args.init_checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.init_checkpoint}")
    if args.steps <= 0 or args.check_every <= 0:
        parser.error("--steps and --check-every must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not math.isfinite(args.target) or args.target <= 0:
        parser.error("--target must be finite and positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    source = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    if not isinstance(source, dict):
        parser.error("checkpoint root must be a mapping")
    dictionary = _load_dictionary(source, device)
    output = args.output or (
        args.init_checkpoint.parent / "pretrained_dictionaries_incoherent.pth"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    initial_mu = float(mutual_coherence_of(dictionary).item())
    started = time.monotonic()
    optimized, history, target_met = optimize_dictionary(
        dictionary,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
        target=args.target,
        check_every=args.check_every,
        max_weight=args.coherence_max_weight,
        max_temperature=args.max_temperature,
    )
    flat = optimized.flatten(end_dim=-2)
    final_mu = float(mutual_coherence_of(optimized).item())
    payload = dict(source)
    payload.update(
        {
            "dictionary": flat.detach().cpu(),
            "expert_dictionary": optimized.detach().cpu(),
            "coherence": {
                **source.get("coherence", {}),
                "warmup_initial_max_off_diagonal": initial_mu,
                "maximum_after_warmup": final_mu,
                "warmup_target": args.target,
                "warmup_target_met": target_met,
            },
            "coherence_warmup": {
                "status": "target_met" if target_met else "target_not_met",
                "initial_mu": initial_mu,
                "final_mu": final_mu,
                "target": args.target,
                "steps": history[-1]["step"] if history else 0,
                "learning_rate": args.learning_rate,
                "weight_decay": 0.0,
                "max_weight": args.coherence_max_weight,
                "max_temperature": args.max_temperature,
                "history": history,
                "elapsed_seconds": time.monotonic() - started,
                "updated_unix": time.time(),
            },
        }
    )
    if "model_state_dict" in payload:
        payload["model_state_dict"]["dual_domain.dictionary"] = optimized.detach().cpu()
    _atomic_save(output, payload)
    report = {
        "output": str(output),
        **payload["coherence_warmup"],
        "dictionary_shape": list(optimized.shape),
        "welch_bound": MutualCoherenceLoss().welch_bound(
            optimized.size(-2), optimized.size(-1)
        ),
    }
    print(json.dumps(report, indent=2), flush=True)
    return 0 if target_met else 2


if __name__ == "__main__":
    raise SystemExit(main())
