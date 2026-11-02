"""Learn a compact unsupervised patch encoder and seed 4 expert dictionaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.micro_loader import SUPPORTED_EXTENSIONS


class PatchAutoencoder(nn.Module):
    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3 * 16 * 16),
            nn.Sigmoid(),
        )

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(patches)
        reconstruction = self.decoder(embedding).view(-1, 3, 16, 16)
        return embedding, reconstruction


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_patches(
    image_dirs: list[Path], patch_count: int, seed: int
) -> tuple[torch.Tensor, list[str]]:
    paths = sorted(
        {
            path.resolve()
            for directory in image_dirs
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        }
    )
    if not paths:
        raise FileNotFoundError("no supported images found in the supplied directories")
    if patch_count <= 0:
        raise ValueError(f"patch_count must be positive; got {patch_count}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    patches = torch.empty((patch_count, 3, 16, 16), dtype=torch.uint8)
    written = 0
    files_used: list[str] = []
    for image_index, path in enumerate(paths):
        remaining = patch_count - written
        if remaining <= 0:
            break
        count = math.ceil(remaining / (len(paths) - image_index))
        with Image.open(path) as source:
            image = source.convert("RGB")
            pixels = torch.frombuffer(
                bytearray(image.tobytes()), dtype=torch.uint8
            ).view(image.height, image.width, 3)
        if pixels.size(0) < 16 or pixels.size(1) < 16:
            continue
        for _ in range(count):
            top = int(torch.randint(pixels.size(0) - 15, (), generator=generator))
            left = int(torch.randint(pixels.size(1) - 15, (), generator=generator))
            patch = pixels[top : top + 16, left : left + 16]
            patches[written] = patch.permute(2, 0, 1)
            written += 1
            if written == patch_count:
                break
        files_used.append(str(path))
    if written < patch_count:
        raise ValueError(
            f"only sampled {written} valid patches; requested {patch_count}. "
            "Provide more or larger source images."
        )
    return patches, files_used


def _cluster_kmeans_plus_plus(
    embeddings: torch.Tensor,
    cluster_count: int,
    device: torch.device,
    seed: int,
    init_sample_count: int,
    iterations: int,
    chunk_size: int,
    checkpoint_path: Path | None = None,
    checkpoint_every_centers: int = 512,
) -> torch.Tensor:
    if embeddings.ndim != 2 or embeddings.size(0) < cluster_count:
        raise ValueError(
            f"need at least {cluster_count} embeddings, got {tuple(embeddings.shape)}"
        )
    if checkpoint_every_centers <= 0 or chunk_size <= 0:
        raise ValueError("checkpoint interval and chunk size must be positive")
    embeddings = embeddings.detach().to(dtype=torch.float32, device="cpu").contiguous()
    fingerprint = hashlib.sha256(
        bytes(embeddings.untyped_storage())
    ).hexdigest()
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    sample_count = min(init_sample_count, embeddings.size(0))
    init_indices = torch.randperm(
        embeddings.size(0), generator=cpu_generator
    )[:sample_count]
    candidates = embeddings.index_select(0, init_indices).to(device)
    centers = torch.empty(
        (cluster_count, embeddings.size(1)), dtype=embeddings.dtype, device=device
    )
    closest = torch.full(
        (candidates.size(0),), float("inf"), device=device, dtype=embeddings.dtype
    )
    next_center = 1
    next_iteration = 0
    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        expected = {
            "cluster_count": cluster_count,
            "embedding_dim": embeddings.size(1),
            "patch_count": embeddings.size(0),
            "seed": seed,
            "init_sample_count": init_sample_count,
            "iterations": iterations,
            "chunk_size": chunk_size,
            "embedding_sha256": fingerprint,
        }
        if checkpoint.get("config") != expected:
            raise ValueError(
                f"K-Means++ resume state {checkpoint_path} does not match "
                "the current embeddings or clustering configuration"
            )
        phase = checkpoint["phase"]
        if phase == "complete":
            return checkpoint["centers"].to(device)
        if phase == "initializing":
            saved_centers = checkpoint["centers"].to(device)
            centers[: saved_centers.size(0)] = saved_centers
            closest = checkpoint["closest"].to(device)
            next_center = int(checkpoint["next_center"])
            generator.set_state(checkpoint["generator_state"])
        elif phase == "lloyd":
            centers = checkpoint["centers"].to(device)
            next_iteration = int(checkpoint["next_iteration"])
        else:
            raise ValueError(f"unknown K-Means++ resume phase: {phase!r}")
    else:
        centers[0] = candidates[
            int(torch.randint(candidates.size(0), (), generator=generator, device=device))
        ]

    def save_state(phase: str, **state: Any) -> None:
        if checkpoint_path is None:
            return
        _atomic_save(
            checkpoint_path,
            {
                "config": {
                    "cluster_count": cluster_count,
                    "embedding_dim": embeddings.size(1),
                    "patch_count": embeddings.size(0),
                    "seed": seed,
                    "init_sample_count": init_sample_count,
                    "iterations": iterations,
                    "chunk_size": chunk_size,
                    "embedding_sha256": fingerprint,
                },
                "phase": phase,
                **state,
            },
        )
        status = {
            "status": "completed" if phase == "complete" else "in_progress",
            "phase": phase,
            "updated_unix": time.time(),
        }
        for key in ("next_center", "next_iteration"):
            if key in state:
                status[key] = state[key]
        status_path = checkpoint_path.with_name("kmeans_status.json")
        temporary = status_path.with_suffix(status_path.suffix + ".tmp")
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        os.replace(temporary, status_path)

    if checkpoint_path is not None and not checkpoint_path.exists():
        save_state(
            "initializing",
            centers=centers[:1].cpu(),
            closest=closest.cpu(),
            next_center=1,
            generator_state=generator.get_state(),
        )

    for index in range(next_center, cluster_count):
        distance = torch.cdist(candidates, centers[index - 1 : index]).squeeze(1)
        closest = torch.minimum(closest, distance.square())
        total = closest.sum()
        if not bool(torch.isfinite(total)) or float(total) <= 0:
            selected = int(torch.randint(candidates.size(0), (), generator=generator, device=device))
        else:
            selected = int(
                torch.multinomial(closest, 1, generator=generator).item()
            )
        centers[index] = candidates[selected]
        if (
            checkpoint_path is not None
            and ((index + 1) % checkpoint_every_centers == 0 or index + 1 == cluster_count)
        ):
            save_state(
                "initializing",
                centers=centers[: index + 1].cpu(),
                closest=closest.cpu(),
                next_center=index + 1,
                generator_state=generator.get_state(),
            )

    data = embeddings.to(device)
    save_state(
        "lloyd",
        centers=centers.cpu(),
        next_iteration=next_iteration,
    )
    for iteration in range(next_iteration, iterations):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(cluster_count, dtype=torch.long, device=device)
        for start in range(0, data.size(0), chunk_size):
            batch = data[start : start + chunk_size]
            labels = torch.cdist(batch, centers).argmin(dim=1)
            sums.index_add_(0, labels, batch)
            counts.index_add_(0, labels, torch.ones_like(labels))
        empty = counts == 0
        centers = sums / counts.clamp_min(1).unsqueeze(1)
        if bool(empty.any()):
            replacements = torch.randperm(
                data.size(0), generator=generator, device=device
            )[: int(empty.sum())]
            centers[empty] = data[replacements]
        save_state(
            "lloyd",
            centers=centers.cpu(),
            next_iteration=iteration + 1,
        )
    centers = F.normalize(centers, p=2, dim=-1)
    save_state("complete", centers=centers.cpu())
    return centers


def _maximum_coherence(dictionary: torch.Tensor) -> float:
    normalized = F.normalize(dictionary, p=2, dim=-1)
    gram = normalized @ normalized.T
    gram.fill_diagonal_(0)
    return float(gram.abs().max().item())


def _save_encoder_training_state(
    path: Path,
    model: PatchAutoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    next_batch: int,
    global_step: int,
    seed: int,
    encoder_config: dict[str, Any],
) -> None:
    state: dict[str, Any] = {
        "encoder_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": global_step,
        "seed": seed,
        "encoder_config": encoder_config,
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    _atomic_save(path, state)
    path.with_name("encoder_status.json").write_text(
        json.dumps(
            {
                "status": "in_progress",
                "epoch": epoch + 1,
                "next_batch": next_batch,
                "global_step": global_step,
                "checkpoint": str(path),
                "updated_unix": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-count", type=int, default=100_000)
    parser.add_argument("--cluster-count", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--encoder-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--kmeans-init-samples", type=int, default=25_000)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--checkpoint-every-centers", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)
    if args.cluster_count < args.embedding_dim:
        parser.error("--cluster-count must be at least --embedding-dim")
    if args.cluster_count % 4:
        parser.error("--cluster-count must be divisible by 4 for expert partitioning")
    if args.patch_count < args.cluster_count:
        parser.error("--patch-count must be at least --cluster-count")
    if args.embedding_dim <= 0 or args.encoder_epochs <= 0:
        parser.error("embedding dimension and encoder epochs must be positive")
    if args.batch_size <= 0 or args.checkpoint_every_steps <= 0:
        parser.error("batch size and checkpoint interval must be positive")
    if args.checkpoint_every_centers <= 0:
        parser.error("--checkpoint-every-centers must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    for directory in args.image_dir:
        if not directory.is_dir():
            parser.error(f"image directory does not exist: {directory}")

    _seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "encoder_latest.pt"
    encoder_status_path = args.output_dir / "encoder_status.json"
    encoder = PatchAutoencoder(args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate)
    encoder_config = {
        "patch_count": args.patch_count,
        "embedding_dim": args.embedding_dim,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "image_dirs": [str(path.resolve()) for path in args.image_dir],
    }
    start_epoch = 0
    first_batch = 0
    global_step = 0
    completed = False
    if latest_path.is_file():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        if state.get("encoder_config") != encoder_config:
            parser.error(
                "existing encoder checkpoint configuration differs from this "
                "run; use its original settings or a new output directory"
            )
        encoder.load_state_dict(state["encoder_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state["epoch"])
        first_batch = int(state["next_batch"])
        global_step = int(state["global_step"])
        rng = state["rng_state"]
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        completed = bool(state.get("encoder_complete", False))
        print(
            f"Resuming patch encoder at epoch={start_epoch + 1}, "
            f"batch={first_batch}, step={global_step}",
            flush=True,
        )

    patches, files_used = _collect_patches(
        args.image_dir, args.patch_count, args.seed
    )
    if not completed:
        dataset = TensorDataset(patches)
        for epoch in range(start_epoch, args.encoder_epochs):
            order = torch.randperm(
                len(dataset), generator=torch.Generator().manual_seed(args.seed + epoch)
            ).tolist()
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=order,
                num_workers=0,
            )
            skip_batches = first_batch if epoch == start_epoch else 0
            first_batch = 0
            encoder.train()
            for batch_index, (patch_batch,) in enumerate(loader):
                if batch_index < skip_batches:
                    continue
                patches_float = patch_batch.to(device=device, dtype=torch.float32) / 255
                _, reconstruction = encoder(patches_float)
                loss = F.mse_loss(reconstruction, patches_float)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                if global_step % args.checkpoint_every_steps == 0:
                    _save_encoder_training_state(
                        latest_path,
                        encoder,
                        optimizer,
                        epoch,
                        batch_index + 1,
                        global_step,
                        args.seed,
                        encoder_config,
                    )
                if global_step % max(1, args.checkpoint_every_steps // 10) == 0:
                    with (args.output_dir / "encoder_steps.jsonl").open(
                        "a", encoding="utf-8"
                    ) as output:
                        output.write(
                            json.dumps(
                                {
                                    "epoch": epoch + 1,
                                    "batch": batch_index + 1,
                                    "global_step": global_step,
                                    "reconstruction_loss": float(loss.detach().item()),
                                }
                            )
                            + "\n"
                        )
            print(
                f"encoder_epoch={epoch + 1}/{args.encoder_epochs} "
                f"reconstruction_loss={float(loss.detach().item()):.6f}",
                flush=True,
            )
            _save_encoder_training_state(
                latest_path,
                encoder,
                optimizer,
                epoch + 1,
                0,
                global_step,
                args.seed,
                encoder_config,
            )
        state = torch.load(latest_path, map_location=device, weights_only=False)
        state["encoder_complete"] = True
        state["epoch"] = args.encoder_epochs
        state["next_batch"] = 0
        _atomic_save(latest_path, state)
        encoder_status_path.write_text(
            json.dumps(
                {
                    "status": "encoder_complete",
                    "epochs": args.encoder_epochs,
                    "global_step": global_step,
                    "updated_unix": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        print("Patch encoder already complete; reusing saved weights.", flush=True)

    encoder.eval()
    embeddings: list[torch.Tensor] = []
    with torch.inference_mode():
        for (patch_batch,) in DataLoader(
            TensorDataset(patches), batch_size=args.batch_size, shuffle=False
        ):
            batch = patch_batch.to(device=device, dtype=torch.float32) / 255
            latent, _ = encoder(batch)
            embeddings.append(F.normalize(latent, p=2, dim=-1).cpu())
    latent_bank = torch.cat(embeddings)
    centers = _cluster_kmeans_plus_plus(
        latent_bank,
        args.cluster_count,
        device,
        args.seed,
        args.kmeans_init_samples,
        args.kmeans_iterations,
        args.chunk_size,
        checkpoint_path=args.output_dir / "kmeans_latest.pt",
        checkpoint_every_centers=args.checkpoint_every_centers,
    ).cpu()
    maximum_coherence = _maximum_coherence(centers)
    welch_bound = math.sqrt(
        (args.cluster_count - args.embedding_dim)
        / (args.embedding_dim * (args.cluster_count - 1))
    )
    checkpoint_path = args.output_dir / "pretrained_dictionaries.pth"
    _atomic_save(
        checkpoint_path,
        {
            "dictionary": centers,
            "expert_dictionary": centers.reshape(
                4, args.cluster_count // 4, args.embedding_dim
            ),
            "encoder_state_dict": encoder.state_dict(),
            "config": {
                "patch_count": args.patch_count,
                "cluster_count": args.cluster_count,
                "embedding_dim": args.embedding_dim,
                "patch_size": 16,
                "expert_count": 4,
                "atoms_per_expert": args.cluster_count // 4,
                "encoder": "PatchAutoencoder",
                "seed": args.seed,
            },
            "coherence": {
                "initial_max_off_diagonal": maximum_coherence,
                "welch_bound": welch_bound,
                "target": 0.15,
                "target_met_at_initialization": maximum_coherence < 0.15,
                "regularization": (
                    "Continue with MutualCoherenceLoss during Stage B and C; "
                    "do not apply Gram-Schmidt to an overcomplete dictionary."
                ),
            },
            "source_images": files_used,
        },
    )
    report = {
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "shape": list(centers.shape),
        "patch_count": len(latent_bank),
        "maximum_initial_coherence": maximum_coherence,
        "welch_bound": welch_bound,
        "target_coherence": 0.15,
        "target_met_at_initialization": maximum_coherence < 0.15,
        "coherence_regularization_required": maximum_coherence >= 0.15,
        "source_image_count": len(files_used),
        "updated_unix": time.time(),
    }
    (args.output_dir / "dictionary_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
