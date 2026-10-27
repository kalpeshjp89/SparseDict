"""Train and validate the MoE Top-K restorer on the DIV2K micro-dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.micro_loader import create_micro_dataloader
from experiments.phase4_utils import gradient_difference_loss, psnr, tiled_predict
from models.moe_topk_restorer import MoETopKRestorer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: MoETopKRestorer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tile_size: int = 128,
    overlap: int = 16,
) -> tuple[float, float]:
    model.eval()
    psnr_values = []
    losses = []
    for corrupted, clean in loader:
        corrupted_image = corrupted[0]
        clean_image = clean[0]
        if corrupted_image.shape[-2:] != clean_image.shape[-2:]:
            corrupted_image = F.interpolate(
                corrupted_image.unsqueeze(0),
                size=clean_image.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(0, 1)
        prediction = tiled_predict(
            model,
            corrupted_image,
            device,
            tile_size=tile_size,
            overlap=overlap,
        )
        losses.append(float(F.l1_loss(prediction, clean_image).item()))
        psnr_values.append(float(psnr(prediction, clean_image).item()))
    return sum(losses) / len(losses), sum(psnr_values) / len(psnr_values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/content/DIV2K"))
    parser.add_argument(
        "--degradation",
        choices=("noise", "super_resolution", "rain"),
        default="noise",
    )
    parser.add_argument("--image-count", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dictionary-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--active-experts", type=int, default=2)
    parser.add_argument("--expert-warmup-epochs", type=int, default=50)
    parser.add_argument("--gradient-weight", type=float, default=0.1)
    parser.add_argument("--validation-tile-size", type=int, default=128)
    parser.add_argument("--validation-overlap", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-psnr", type=float, default=29.5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "phase4")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (defaults to CUDA when available, otherwise CPU)",
    )
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.image_count <= 0:
        parser.error("--image-count must be positive")
    if args.patch_size <= 0:
        parser.error("--patch-size must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.expert_warmup_epochs < 0:
        parser.error("--expert-warmup-epochs must be non-negative")
    if args.gradient_weight < 0:
        parser.error("--gradient-weight must be non-negative")
    if (
        args.validation_tile_size <= 0
        or args.validation_overlap < 0
        or args.validation_overlap >= args.validation_tile_size
    ):
        parser.error(
            "require --validation-tile-size>0 and "
            "0<=--validation-overlap<--validation-tile-size"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but no CUDA device is available")
    set_seed(args.seed)

    train_loader = create_micro_dataloader(
        args.data_root,
        args.degradation,
        image_count=args.image_count,
        batch_size=1,
        seed=args.seed,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        cache_noise=False,
        patch_size=args.patch_size,
    )
    validation_loader = create_micro_dataloader(
        args.data_root,
        args.degradation,
        image_count=args.image_count,
        batch_size=1,
        seed=args.seed + 10_000,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        cache_noise=True,
    )

    model = MoETopKRestorer(
        d=args.dim,
        m=args.dictionary_size,
        active_experts=args.active_experts,
        top_k=args.top_k,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(args.epochs):
        train_loader.dataset.set_epoch(epoch)
        active_experts = (
            args.active_experts if epoch < args.expert_warmup_epochs else 1
        )
        model.router.active_experts = active_experts
        model.train()
        total_loss = 0.0
        total_reconstruction = 0.0
        total_gradient = 0.0
        for corrupted, clean in train_loader:
            corrupted = corrupted.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, auxiliary_loss = model(
                corrupted, output_size=clean.shape[-2:], return_aux=True
            )
            reconstruction_loss = F.l1_loss(prediction, clean)
            gradient_loss = gradient_difference_loss(prediction, clean)
            loss = (
                reconstruction_loss
                + args.gradient_weight * gradient_loss
                + auxiliary_loss
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
            total_reconstruction += float(reconstruction_loss.detach().item())
            total_gradient += float(gradient_loss.detach().item())

        record = {
            "epoch": epoch + 1,
            "active_experts": active_experts,
            "train_loss": total_loss / len(train_loader),
            "train_l1": total_reconstruction / len(train_loader),
            "train_gradient_loss": total_gradient / len(train_loader),
        }
        if (epoch + 1) % max(1, args.epochs // 20) == 0 or epoch + 1 == args.epochs:
            validation_l1, validation_psnr = evaluate(
                model,
                validation_loader,
                device,
                tile_size=args.validation_tile_size,
                overlap=args.validation_overlap,
            )
            record["validation_l1"] = validation_l1
            record["validation_psnr"] = validation_psnr
            print(
                f"epoch={epoch + 1:03d}/{args.epochs} "
                f"train_l1={record['train_l1']:.6f} "
                f"validation_l1={validation_l1:.6f} "
                f"validation_psnr={validation_psnr:.3f} dB",
                flush=True,
            )
        history.append(record)

    _, final_psnr = evaluate(
        model,
        validation_loader,
        device,
        tile_size=args.validation_tile_size,
        overlap=args.validation_overlap,
    )
    passed = final_psnr >= args.min_psnr
    checkpoint_path = args.output_dir / "moe_topk_restorer.pt"
    metrics_path = args.output_dir / "metrics.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "d": args.dim,
                "m": args.dictionary_size,
                "num_experts": 4,
                "active_experts": model.router.active_experts,
                "top_k": args.top_k,
                "degradation": args.degradation,
            },
            "seed": args.seed,
            "epochs": args.epochs,
            "final_validation_psnr": final_psnr,
        },
        checkpoint_path,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "degradation": args.degradation,
                "epochs": args.epochs,
                "image_count": args.image_count,
                "patch_size": args.patch_size,
                "expert_warmup_epochs": args.expert_warmup_epochs,
                "gradient_weight": args.gradient_weight,
                "validation_tile_size": args.validation_tile_size,
                "validation_overlap": args.validation_overlap,
                "device": str(device),
                "min_psnr": args.min_psnr,
                "final_validation_psnr": final_psnr,
                "pass": passed,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"checkpoint={checkpoint_path}")
    print(f"metrics={metrics_path}")
    print(
        f"final validation PSNR={final_psnr:.3f} dB "
        f"target={args.min_psnr:.3f} dB pass={passed}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
