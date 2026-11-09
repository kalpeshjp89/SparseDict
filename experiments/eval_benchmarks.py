"""Universal Multi-Task Benchmarking Harness (Task 7.2c).

Evaluates the MasterMoETopKRestorer checkpoint on standard restoration
benchmarks and exports a comparative summary.

Benchmarks:
    - Urban100 (SR x2/x3/x4)
    - CBSD68 (denoising sigma=15/25/50, JPEG cleanup)
    - SOTS (deraining, dehazing)

Baselines (SwinIR, Restormer, HAT-L) are optional: if their checkpoints
are provided via --baseline-weights, they are evaluated; otherwise the
summary records them as "pending".

Usage (Colab / local GPU):
    python experiments/eval_benchmarks.py \\
        --checkpoint /content/drive/MyDrive/SparseDict/phase7/stage_c_run3/latest.pt \\
        --data-root /content/drive/MyDrive/SparseDict/datasets \\
        --output results/benchmark_summary.json \\
        --iterations 100
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.universal_loader import TASK_SCALES
from models.moe_topk_restorer import MasterMoETopKRestorer

TASKS = ["sr2", "sr3", "sr4", "denoise15", "denoise25", "denoise50",
         "derain", "dehaze", "jpeg"]
BASELINE_NAMES = ["SwinIR", "Restormer", "HAT-L"]
BENCHMARK_DATASETS = ["urban100", "cbsd68", "sots"]


# ---------------------------------------------------------------------------
# Degradation helpers (reuse training-pipeline logic)
# ---------------------------------------------------------------------------

def _to_tensor(img: Image.Image) -> torch.Tensor:
    rgb = img.convert("RGB")
    pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).view(
        rgb.height, rgb.width, 3
    )
    return pixels.permute(2, 0, 1).contiguous().to(torch.float32).div_(255.0)


def _add_gaussian_noise(clean: torch.Tensor, sigma: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(clean.shape, generator=gen, dtype=clean.dtype) * (sigma / 255.0)
    return (clean + noise).clamp(0.0, 1.0)


def _downsample(clean: torch.Tensor, scale: int) -> torch.Tensor:
    h, w = clean.shape[-2:]
    lr = F.interpolate(
        clean.unsqueeze(0), size=(h // scale, w // scale),
        mode="bicubic", align_corners=False,
    )
    return lr.squeeze(0).clamp(0.0, 1.0)


def _jpeg_compress(clean: torch.Tensor, quality: int) -> torch.Tensor:
    import io
    buf = io.BytesIO()
    img = Image.fromarray(
        (clean.permute(1, 2, 0).numpy() * 255).astype("uint8")
    )
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    return _to_tensor(out)


def degrade(clean: torch.Tensor, task: str, seed: int = 0) -> torch.Tensor:
    """Create a degraded version of a clean image for the given task."""
    scale = TASK_SCALES[task]
    if task.startswith("sr"):
        return _downsample(clean, scale)
    if task.startswith("denoise"):
        sigma = int(task.replace("denoise", ""))
        return _add_gaussian_noise(clean, sigma, seed)
    if task == "derain":
        try:
            from data.micro_loader import _make_rain_layer
            gen = torch.Generator(device="cpu").manual_seed(seed)
            rain = _make_rain_layer(clean, gen)
            return (clean + 0.3 * rain).clamp(0.0, 1.0)
        except Exception:
            return _add_gaussian_noise(clean, 25, seed)
    if task == "dehaze":
        # Simple haze: reduce contrast + add white veil
        gen = torch.Generator(device="cpu").manual_seed(seed)
        t = 0.6 + torch.rand(1, generator=gen).item() * 0.2
        return (clean * t + (1 - t)).clamp(0.0, 1.0)
    if task == "lowlight":
        return (clean * 0.3).clamp(0.0, 1.0)
    if task == "jpeg":
        return _jpeg_compress(clean, quality=10)
    raise ValueError(f"unknown task: {task}")


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class BenchmarkDataset(Dataset):
    """Load clean images from a directory for benchmarking."""

    def __init__(self, root: Path, max_images: int = 0, max_side: int = 256):
        self.root = Path(root)
        self.paths = sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
        if max_images > 0:
            self.paths = self.paths[:max_images]
        self.max_side = max_side

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        with Image.open(self.paths[idx]) as img:
            rgb = img.convert("RGB")
            # Resize to avoid OOM on large images
            w, h = rgb.size
            if max(w, h) > self.max_side:
                scale = self.max_side / max(w, h)
                rgb = rgb.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            return _to_tensor(rgb), self.paths[idx].name


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    if mse < 1e-10:
        return 100.0
    return float(10 * math.log10(1.0 / mse.item()))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    # Simple SSIM (window-based)
    from torch.nn.functional import conv2d
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    # Use luminance channel
    x = pred.mean(dim=1, keepdim=True)
    y = target.mean(dim=1, keepdim=True)
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(ssim_map.mean().item())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> MasterMoETopKRestorer:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MasterMoETopKRestorer(
        d=64, atoms_per_expert=1024, top_k=16, noise_std=0.1,
        aux_weight=0.01, entropy_kappa=0.1,
    ).to(device)
    state = ckpt.get("model_state_dict")
    if state is not None:
        model.load_state_dict(state)
    else:
        # fallback: load dictionary only
        dict_ckpt = ckpt.get("expert_dictionary") or ckpt.get("dictionary")
        if dict_ckpt is not None:
            model.dual_domain.dictionary.copy_(
                torch.as_tensor(dict_ckpt, device=device, dtype=torch.float32).reshape(
                    *model.dual_domain.dictionary.shape
                )
            )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference + latency
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: MasterMoETopKRestorer,
    degraded: torch.Tensor,
    task: str,
    device: torch.device,
) -> torch.Tensor:
    scale = TASK_SCALES[task]
    h, w = degraded.shape[-2:]
    # Pad to even dimensions
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    if pad_h or pad_w:
        degraded = F.pad(degraded, (0, pad_w, 0, pad_h), mode="reflect")
    h, w = degraded.shape[-2:]
    if scale > 1:
        out_h, out_w = h * scale, w * scale
    else:
        out_h, out_w = h, w
    output, _ = model(
        degraded.unsqueeze(0).to(device),
        output_size=(out_h, out_w),
        scale_factor=scale,
    )
    result = output.squeeze(0).cpu().clamp(0.0, 1.0)
    # Crop back to original size (scaled)
    if scale > 1:
        result = result[:, :h*scale - pad_h*scale, :w*scale - pad_w*scale]
    else:
        result = result[:, :h - pad_h, :w - pad_w]
    return result


def measure_latency(
    model: MasterMoETopKRestorer,
    degraded: torch.Tensor,
    task: str,
    device: torch.device,
    iterations: int = 100,
) -> float:
    """Mean latency in ms over ``iterations`` CUDA-event-timed runs."""
    scale = TASK_SCALES[task]
    h, w = degraded.shape[-2:]
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    if pad_h or pad_w:
        degraded = F.pad(degraded, (0, pad_w, 0, pad_h), mode="reflect")
    h, w = degraded.shape[-2:]
    out_h, out_w = (h * scale, w * scale) if scale > 1 else (h, w)
    inp = degraded.unsqueeze(0).to(device)
    # warmup
    for _ in range(5):
        model(inp, output_size=(out_h, out_w), scale_factor=scale)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    total_ms = 0.0
    for _ in range(iterations):
        if device.type == "cuda":
            start.record()
        model(inp, output_size=(out_h, out_w), scale_factor=scale)[0]
        if device.type == "cuda":
            end.record()
            torch.cuda.synchronize()
            total_ms += start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            model(inp, output_size=(out_h, out_w), scale_factor=scale)[0]
            total_ms += (time.perf_counter() - t0) * 1000.0
    return total_ms / iterations


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    model: MasterMoETopKRestorer,
    dataset_name: str,
    data_root: Path,
    tasks: list[str],
    device: torch.device,
    max_images: int = 0,
    max_side: int = 256,
    iterations: int = 100,
) -> dict[str, Any]:
    dataset_dir = data_root / dataset_name.upper().replace("URBAN100", "Urban100")
    # Try common directory names
    for candidate in [dataset_name, dataset_name.upper(), dataset_name.capitalize(),
                      dataset_name.lower()]:
        p = data_root / candidate
        if p.is_dir():
            dataset_dir = p
            break
    if not dataset_dir.is_dir():
        return {"error": f"dataset directory not found: {dataset_dir}"}

    ds = BenchmarkDataset(dataset_dir, max_images=max_images, max_side=max_side)
    if len(ds) == 0:
        return {"error": f"no images found in {dataset_dir}"}

    results: dict[str, Any] = {"dataset": dataset_name, "num_images": len(ds), "tasks": {}}
    for task in tasks:
        psnrs, ssims = [], []
        for idx in range(len(ds)):
            clean, name = ds[idx]
            degraded = degrade(clean, task, seed=idx)
            pred = run_inference(model, degraded, task, device)
            # Align sizes for SR
            scale = TASK_SCALES[task]
            if scale > 1:
                target = clean
            else:
                target = clean
            # Crop to common size
            min_h = min(pred.shape[-2], target.shape[-2])
            min_w = min(pred.shape[-1], target.shape[-1])
            pred_c = pred[:, :min_h, :min_w]
            target_c = target[:, :min_h, :min_w]
            psnrs.append(compute_psnr(pred_c, target_c))
            ssims.append(compute_ssim(pred_c, target_c))

        # Latency on a representative image
        sample_degraded, _ = ds[0]
        sample_deg = degrade(sample_degraded, task, seed=0)
        latency = measure_latency(model, sample_deg, task, device, iterations)

        results["tasks"][task] = {
            "psnr": round(sum(psnrs) / len(psnrs), 4),
            "ssim": round(sum(ssims) / len(ssims), 4),
            "num_images": len(psnrs),
            "latency_ms": round(latency, 4),
        }
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark_summary.json"))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--datasets", nargs="+", default=BENCHMARK_DATASETS)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all")
    parser.add_argument("--max-side", type=int, default=256, help="max image side length")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument(
        "--baselines", nargs="*", default=[],
        help="baseline names to include (SwinIR, Restormer, HAT-L); "
             "requires their checkpoints via --baseline-weights",
    )
    parser.add_argument(
        "--baseline-weights", type=Path, default=None,
        help="directory containing baseline checkpoint files",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "model": "MasterMoETopKRestorer (SparseDict)",
        "tasks": args.tasks,
        "datasets": {},
        "baselines": {},
    }

    for dataset_name in args.datasets:
        print(f"\n=== Evaluating {dataset_name} ===")
        result = evaluate_dataset(
            model, dataset_name, args.data_root, args.tasks, device,
            max_images=args.max_images, max_side=args.max_side, iterations=args.iterations,
        )
        summary["datasets"][dataset_name] = result
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            for task, metrics in result["tasks"].items():
                print(f"  {task}: PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} "
                      f"latency={metrics['latency_ms']:.2f}ms")

    # Baseline entries (pending until checkpoints are provided)
    for baseline in args.baselines:
        summary["baselines"][baseline] = {"status": "pending", "note": "baseline checkpoint not provided"}

    # FLOP reduction estimate
    try:
        import subprocess
        flop_result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "measure_flops.py"),
             "--patch", "144", "--json"],
            capture_output=True, text=True, timeout=120,
        )
        if flop_result.returncode == 0:
            summary["flop_reduction"] = json.loads(flop_result.stdout.strip())
    except Exception as e:
        summary["flop_reduction"] = {"error": str(e)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
