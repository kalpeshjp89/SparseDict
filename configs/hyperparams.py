"""Phase-1 hyperparameters with an explicit reason for every value.

Values are centralized so later phases can retune without hunting magic numbers.
Each entry records: value, why it exists now, and which spec it came from.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hyperparam:
    value: float | int | bool | str
    rationale: str
    source: str


# --- tensor geometry (architecture overview §2, Phase-0 toy scale) ---

D = Hyperparam(
    value=64,
    rationale="Lightweight hidden size from the spec; keeps toy runs on CPU while matching the 'lightweight' column.",
    source="01_architecture_overview.md §2 Hidden Dimension",
)

REDUCTION_RATIO = Hyperparam(
    value=4,
    rationale="Key/query bottleneck: d_k = d / r. Cuts affinity cost by 4x vs full-d attention without changing output dim.",
    source="01_architecture_overview.md §2; todo Task 1.1",
)

M = Hyperparam(
    value=256,
    rationale="Phase-0 toy dictionary, not the full 4096-atom bank. Fail-fast: enough atoms to exercise softmax over M, cheap enough for 50 CPU steps.",
    source="todo.md Phase 0 Task 0.1",
)

N = Hyperparam(
    value=64,
    rationale="Token count for the smoke test. Full 64x64-patch N=4096 is a later latency target; 64 tokens still gives a real B×N×M affinity.",
    source="fail-fast vs 01_architecture_overview.md N=4096 example",
)

B = Hyperparam(
    value=2,
    rationale="Minimum batch that still exercises batched matmul (not a singleton that hides broadcast bugs).",
    source="todo.md Phase 0 Task 0.1",
)

# --- temperature (module spec 03 §2) ---

TAU_0 = Hyperparam(
    value=1.0,
    rationale="Start of the documented cosine anneal (1.0 → 0.05). Phase 1 is un-sparsified dense softmax; keep tau_0 at the training start so logits are not over-sharpened yet.",
    source="03_topk_sparsification_gradients.md §2",
)

M_0 = Hyperparam(
    value=256,
    rationale="Reference dictionary size in tau = tau_0 * log(M_active)/log(M_0). Set equal to the toy M so tau=1.0 now; when M grows to 4096, tau scales automatically to log(4096)/log(256)=1.5.",
    source="03_topk_sparsification_gradients.md §2; chosen M_0 = current toy M",
)

# --- projections ---

USE_BIAS = Hyperparam(
    value=False,
    rationale="Standard Transformer Q/K/V projections omit bias; extra bias on keys would add a query-independent offset to every atom score.",
    source="Vaswani et al. convention; not specified in SparseDict notes so defaulted off",
)

# --- Task 1.2 optimization smoke test ---

LR = Hyperparam(
    value=1e-3,
    rationale="Adam default-ish step size for a tiny MSE reconstruction. Large enough that 50 steps should move loss; small enough to avoid exploding the dense M-way softmax.",
    source="engineering default for toy overfit; not in the specs",
)

STEPS = Hyperparam(
    value=50,
    rationale="Exact verification budget from Task 1.2.",
    source="todo.md Task 1.2",
)

SEED = Hyperparam(
    value=0,
    rationale="Reproducible toy run so analysis numbers can be compared after hyperparameter changes.",
    source="engineering default",
)

WEIGHT_DECAY = Hyperparam(
    value=0.0,
    rationale="Smoke test is capacity/overfit, not generalization. Decay would fight the 'loss must drop' check.",
    source="engineering default for overfit test",
)


def d_k() -> int:
    return int(D.value) // int(REDUCTION_RATIO.value)


def as_table() -> str:
    rows = []
    for name, obj in globals().items():
        if isinstance(obj, Hyperparam):
            rows.append((name, obj.value, obj.source, obj.rationale))
    lines = [
        "| name | value | source | rationale |",
        "|---|---|---|---|",
    ]
    for name, value, source, rationale in rows:
        lines.append(f"| `{name}` | `{value}` | {source} | {rationale} |")
    lines.append(f"| `d_k` (derived) | `{d_k()}` | d / r | Query/key width after reduction. |")
    return "\n".join(lines)
