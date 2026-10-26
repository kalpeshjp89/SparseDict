"""Phase 1 runner: Task 1.1 shape check + Task 1.2 50-step overfit.

Writes JSON metrics under experiments/results/ for the analysis notes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import hyperparams as hp
from models.modules.dict_attention import DenseDictCrossAttention


RESULTS = ROOT / "experiments" / "results"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def task_1_1() -> dict:
    set_seed(int(hp.SEED.value))
    model = DenseDictCrossAttention()
    x = torch.randn(int(hp.B.value), int(hp.N.value), int(hp.D.value))
    with torch.no_grad():
        k_d, v_d = model.project_dictionary()
        y = model(x)

    expected = (int(hp.B.value), int(hp.N.value), int(hp.D.value))
    ok = tuple(y.shape) == expected
    tau = model.temperature()
    return {
        "pass": ok,
        "x_shape": list(x.shape),
        "x_out_shape": list(y.shape),
        "expected_x_out_shape": list(expected),
        "k_d_shape": list(k_d.shape),
        "v_d_shape": list(v_d.shape),
        "d": model.d,
        "d_k": model.d_k,
        "r": model.reduction_ratio,
        "M": model.m,
        "tau": tau,
        "tau_0": model.tau_0,
        "M_0": model.m_0,
        "param_count": sum(p.numel() for p in model.parameters()),
        "dictionary_params": model.dictionary.numel(),
        "hyperparams": hp.as_table(),
    }


def task_1_2() -> dict:
    set_seed(int(hp.SEED.value))
    model = DenseDictCrossAttention()
    x = torch.randn(int(hp.B.value), int(hp.N.value), int(hp.D.value))
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(hp.LR.value),
        weight_decay=float(hp.WEIGHT_DECAY.value),
    )

    losses: list[float] = []
    for _ in range(int(hp.STEPS.value)):
        opt.zero_grad(set_to_none=True)
        y = model(x)
        loss = F.mse_loss(y, x)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    deltas = [losses[i + 1] - losses[i] for i in range(len(losses) - 1)]
    strictly_decreasing = all(d < 0 for d in deltas)
    n_up = sum(1 for d in deltas if d >= 0)
    return {
        "pass": strictly_decreasing and losses[-1] < losses[0],
        "strictly_decreasing": strictly_decreasing,
        "steps_that_did_not_drop": n_up,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "relative_drop": (losses[0] - losses[-1]) / losses[0] if losses[0] != 0 else None,
        "losses": losses,
        "lr": float(hp.LR.value),
        "steps": int(hp.STEPS.value),
        "seed": int(hp.SEED.value),
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    r1 = task_1_1()
    r2 = task_1_2()
    (RESULTS / "task_1_1.json").write_text(json.dumps(r1, indent=2), encoding="utf-8")
    (RESULTS / "task_1_2.json").write_text(json.dumps(r2, indent=2), encoding="utf-8")

    print("=== Task 1.1 shape check ===")
    print(
        f"X {r1['x_shape']} -> X_out {r1['x_out_shape']} "
        f"(expected {r1['expected_x_out_shape']})  pass={r1['pass']}"
    )
    print(f"K_D {r1['k_d_shape']}  V_D {r1['v_d_shape']}  tau={r1['tau']:.6f}")
    print(f"params={r1['param_count']}  dictionary={r1['dictionary_params']}")

    print("=== Task 1.2 50-step overfit ===")
    print(
        f"loss {r2['loss_first']:.6f} -> {r2['loss_last']:.6f}  "
        f"rel_drop={r2['relative_drop']:.4f}  "
        f"strictly_decreasing={r2['strictly_decreasing']}  pass={r2['pass']}"
    )
    if not r2["strictly_decreasing"]:
        print(f"non-decreasing steps: {r2['steps_that_did_not_drop']}")

    ok = r1["pass"] and r2["pass"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
