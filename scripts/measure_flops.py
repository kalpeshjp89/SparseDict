"""Measure approximate FLOPs of MasterMoETopKRestorer for Stage B training configs.

Uses ``torch.profiler`` (``with_flops=True``) on the torch-composed path
(``use_fused_kernel=False``), which is the same graph the GPU training runs on
Colab (the Triton options are unavailable on T4/CPU).

Counts forward-pass FLOPs of registered aten ops (matmul/conv/etc.). Custom
fused kernels, if enabled, do the same math; profiler occurrences there are
undercounted on purpose. Backward is conventionally ~2x forward for
matmul-heavy graphs; the script prints both the measured forward total and the
backward estimate so consumers can state inputs honestly.

Usage (repo root):
    python scripts/measure_flops.py --patch 144
    python scripts/measure_flops.py --patch 144 --atoms-per-expert 1024 --dim 64 --top-k 16
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.moe_topk_restorer import MasterMoETopKRestorer  # noqa: E402

SCALE_TASKS = {1: 8, 2: 1, 3: 1, 4: 1}  # task mix: 8 s=1 tasks, sr2, sr3, sr4


def count_forward_flops(
    model: MasterMoETopKRestorer, low_size: int, output_size: int, device: str
) -> float:
    """FLOPs for feeding a low_quality patch of side ``low_size`` producing ``output_size``."""
    image = torch.rand(1, 3, low_size, low_size, device=device)
    model.eval()
    with torch.no_grad(), profile(
        activities=[ProfilerActivity.CPU],
        with_flops=True,
        record_shapes=False,
    ) as prof:
        model(image, output_size=(output_size, output_size))
    total = 0.0
    for evt in prof.key_averages():
        total += float(getattr(evt, "flops", 0) or 0)
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", type=int, default=144, help="clean-patch side P")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--atoms-per-expert", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument(
        "--json", action="store_true", help="machine-readable single-line JSON"
    )
    args = parser.parse_args(argv)

    if args.patch % 12:
        raise SystemExit(
            f"--patch must be divisible by 12 (covers sr2/sr3/sr4); got {args.patch}"
        )
    if args.device != "cpu":
        raise SystemExit("this script measures a CPU profile; FLOP counts are device-independent")

    torch.manual_seed(2027)
    model = MasterMoETopKRestorer(
        d=args.dim,
        atoms_per_expert=args.atoms_per_expert,
        top_k=args.top_k,
        noise_std=0.1,
    ).to(args.device)

    scale_gb = {}
    weighted = 0.0
    total_tasks = sum(SCALE_TASKS.values())
    flops_by_op: defaultdict[str, float] = defaultdict(float)

    for scale, n_tasks in SCALE_TASKS.items():
        low = args.patch // scale
        _, stats = None, count_forward_flops(
            model, low, args.patch, args.device
        )
        scale_gb[scale] = stats / 1e9
        weighted += stats * n_tasks / total_tasks

        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU],
            with_flops=True,
        ) as prof:
            image = torch.rand(1, 3, low, low, device=args.device)
            with torch.no_grad():
                model(image, output_size=(args.patch, args.patch))
        for evt in prof.key_averages():
            if getattr(evt, "flops", 0):
                flops_by_op[evt.key] += float(evt.flops)

    backward_est = 2.0  # backward ~2x forward for matmul-dominated graphs

    result = {
        "patch": args.patch,
        "dim": args.dim,
        "atoms_per_expert": args.atoms_per_expert,
        "top_k": args.top_k,
        "forward_GFLOPs_by_scale": {k: round(v, 3) for k, v in scale_gb.items()},
        "forward_GFLOPs_weighted": round(weighted, 3),
        "step_forward_plus_backward_GFLOPs_est": round(
            weighted * (1.0 + backward_est), 3
        ),
        "backward_multiplier_assumed": backward_est,
        "top_flop_ops": {
            k: round(v / 1e9, 3)
            for k, v in sorted(flops_by_op.items(), key=lambda kv: -kv[1])[:6]
        },
    }
    if args.json:
        print(json.dumps(result))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
