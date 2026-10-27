"""Run tiled inference with a trained Phase 4 MoE Top-K checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.micro_loader import create_micro_dataloader
from experiments.phase4_utils import psnr, tiled_predict
from models.moe_topk_restorer import MoETopKRestorer


def _save_rgb_tensor(image: torch.Tensor, path: Path) -> None:
    pixels = (
        image.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
    )
    height, width, _ = pixels.shape
    Image.frombytes("RGB", (width, height), bytes(pixels.untyped_storage())).save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/content/DIV2K"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--degradation",
        choices=("noise", "super_resolution", "rain"),
        default=None,
        help="defaults to the degradation stored in the checkpoint",
    )
    parser.add_argument("--image-count", type=int, default=10)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "phase4" / "inference")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but no CUDA device is available")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_config = checkpoint["model_config"]
    degradation = args.degradation or model_config["degradation"]
    model = MoETopKRestorer(
        d=model_config["d"],
        m=model_config["m"],
        num_experts=model_config["num_experts"],
        active_experts=model_config["active_experts"],
        top_k=model_config["top_k"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = create_micro_dataloader(
        args.data_root,
        degradation,
        image_count=args.image_count,
        batch_size=1,
        shuffle=False,
        cache_noise=degradation == "noise",
        cache_rain=degradation == "rain",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (corrupted, clean) in enumerate(loader):
        corrupted_image = corrupted[0]
        clean_image = clean[0]
        if corrupted_image.shape[-2:] != clean_image.shape[-2:]:
            corrupted_image = F.interpolate(
                corrupted_image.unsqueeze(0),
                size=clean_image.shape[-2:],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(0, 1)
        start = time.perf_counter()
        prediction = tiled_predict(
            model,
            corrupted_image,
            device,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
        elapsed = time.perf_counter() - start
        score = float(psnr(prediction.unsqueeze(0), clean_image.unsqueeze(0)).item())
        output_path = args.output_dir / f"{index + 1:04d}_restored.png"
        _save_rgb_tensor(prediction.cpu(), output_path)
        records.append(
            {
                "image": loader.dataset.image_paths[index].name,
                "output": str(output_path),
                "psnr": score,
                "seconds": elapsed,
            }
        )
        print(
            f"{records[-1]['image']}: PSNR={score:.3f} dB, "
            f"time={elapsed:.2f}s, output={output_path}",
            flush=True,
        )

    summary = {
        "degradation": degradation,
        "image_count": len(records),
        "mean_psnr": sum(record["psnr"] for record in records) / len(records),
        "device": str(device),
        "results": records,
    }
    summary_path = args.output_dir / "inference_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"mean PSNR={summary['mean_psnr']:.3f} dB")
    print(f"metrics={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
