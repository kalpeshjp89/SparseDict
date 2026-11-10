"""Pre-debugging verification script for MasterMoETopKRestorer.

Runs a single batch through the model and prints diagnostic metrics to
validate model sanity before spending GPU cycles on large training runs.

Usage (repo root):
    python scripts/debug_diagnostics.py --checkpoint path/to/latest.pt --device cuda
    python scripts/debug_diagnostics.py --checkpoint path/to/latest.pt --device cuda --manifest path/to/train.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.moe_topk_restorer import MasterMoETopKRestorer


def audit_model_behavior(model, x, y, device="cuda"):
    """Run diagnostic checks on a single batch."""
    model.eval()
    x = x.to(device)
    y = y.to(device) if y is not None else None

    with torch.no_grad():
        # 1. Check Output Range & Residual Magnitude
        pred, losses = model(x, reference=y if y is not None else x, total_loss=False)
        residual_norm = torch.norm(pred - x).item()
        target_residual_norm = torch.norm(y - x).item() if y is not None else 0.0
        print(f"[Check 1] Pred Residual Norm: {residual_norm:.4f} | "
              f"Target Residual Norm: {target_residual_norm:.4f}")
        print(f"           Output range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")

        # 2. Check Entropy Gate Active Ratio
        features = model.stem(x)
        tokens = features.flatten(2).transpose(1, 2)
        gate_result = model.entropy_gate(tokens)
        active_ratio = (gate_result.gate > 0).float().mean().item()
        print(f"[Check 2] Entropy Gate Active Ratio (Tokens reaching Dict): "
              f"{active_ratio * 100:.2f}%")
        if active_ratio < 0.40:
            print("           WARNING: active ratio < 40% — threshold too aggressive, "
                  "texture/edge tokens skipping dictionary lookup")

        # 3. Expert Load Distribution
        attention_map, gates = model.dual_domain(
            features, return_gates=True, scale_factor=1
        )
        routing_load = gates.detach().mean(dim=(0, 1))
        print(f"[Check 3] Expert Load Distribution: "
              f"{[round(float(v), 4) for v in routing_load]}")
        max_load = float(routing_load.max().item())
        if max_load > 0.60:
            print(f"           WARNING: one expert takes {max_load*100:.1f}% of tokens "
                  f"— routing may be collapsing; increase aux_weight")
        print(f"           Aux Load Balance Loss: {model.balance_loss(gates).item():.6f}")

        # 4. Dictionary Coherence State
        coh_loss = model.coherence_loss(model.dual_domain.dictionary)
        print(f"[Check 4] Coherence Loss: {coh_loss.item():.6f}")
        # Compute per-bank mu
        try:
            from scripts.warmup_coherence import mutual_coherence_of
            mu = mutual_coherence_of(model.dual_domain.dictionary.detach())
            print(f"           Mutual Coherence (per-bank max): {mu:.6f}")
        except Exception as e:
            print(f"           (could not compute mu: {e})")

        # 5. Cross-Attention Weight & Norm Audit
        attention_tokens = attention_map.flatten(2).transpose(1, 2)
        qk_norm = torch.norm(attention_tokens).item()
        print(f"[Check 5] Attention output norm: {qk_norm:.4f}")
        print(f"           Attention entropy (mean): {gate_result.entropy.mean().item():.4f}")

        # 6. Spatial-Frequency Residual Energy
        residual = pred - x  # residual magnitude
        print(f"[Check 6] Residual energy (pred - input): {torch.norm(residual).item():.4f}")

    print("\n--- Audit Complete ---")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--manifest", type=str, default=None,
                        help="path to train.csv for a real batch")
    parser.add_argument("--patch-size", type=int, default=64)
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = MasterMoETopKRestorer(
        d=64, atoms_per_expert=1024, top_k=16, noise_std=0.1,
        aux_weight=0.01, entropy_kappa=0.1,
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict")
        if state is not None:
            model.load_state_dict(state)
        else:
            dict_ckpt = ckpt.get("expert_dictionary") or ckpt.get("dictionary")
            if dict_ckpt is not None:
                model.dual_domain.dictionary.copy_(
                    torch.as_tensor(dict_ckpt, device=device, dtype=torch.float32).reshape(
                        *model.dual_domain.dictionary.shape
                    )
                )
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Get a batch
    if args.manifest:
        from data.universal_loader import UniversalRestorationDataset
        ds = UniversalRestorationDataset(args.manifest, patch_size=args.patch_size)
        sample = ds[0]
        x = sample.low_quality.unsqueeze(0)
        y = sample.clean.unsqueeze(0)
    else:
        # Synthetic batch
        x = torch.rand(1, 3, args.patch_size, args.patch_size)
        y = x.clone()

    audit_model_behavior(model, x, y, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
