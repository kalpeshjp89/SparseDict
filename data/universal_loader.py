"""Deterministic synthetic multi-task restoration patches from clean images."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from data.micro_loader import SUPPORTED_EXTENSIONS, _make_rain_layer

TASK_SCALES = {
    "sr2": 2,
    "sr3": 3,
    "sr4": 4,
    "denoise15": 1,
    "denoise25": 1,
    "denoise50": 1,
    "derain": 1,
    "desnow": 1,
    "dehaze": 1,
    "lowlight": 1,
    "jpeg": 1,
}


class RestorationSample(NamedTuple):
    low_quality: torch.Tensor
    clean: torch.Tensor
    task: str
    scale_factor: int


class UniversalRestorationDataset(Dataset[RestorationSample]):
    """Read ``image_path,task`` rows and generate reproducible paired crops."""

    def __init__(
        self,
        manifest: str | Path,
        patch_size: int = 128,
        seed: int = 0,
    ) -> None:
        self.manifest = Path(manifest)
        self.patch_size = int(patch_size)
        self.seed = int(seed)
        self.epoch = 0
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive; got {patch_size}")
        if not self.manifest.is_file():
            raise FileNotFoundError(f"manifest does not exist: {self.manifest}")
        with self.manifest.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        if not rows or "image_path" not in rows[0] or "task" not in rows[0]:
            raise ValueError(
                f"{self.manifest} must have image_path and task CSV columns"
            )
        self.rows: list[tuple[Path, str, Path | None]] = []
        for line, row in enumerate(rows, start=2):
            image_path = Path(row["image_path"]).expanduser()
            if not image_path.is_absolute():
                image_path = self.manifest.parent / image_path
            task = row["task"].strip().lower()
            if task not in TASK_SCALES:
                raise ValueError(f"unsupported task {task!r} at {self.manifest}:{line}")
            if self.patch_size % TASK_SCALES[task]:
                raise ValueError(
                    f"patch_size={self.patch_size} must be divisible by the "
                    f"{TASK_SCALES[task]}x scale for {task}"
                )
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"image from manifest does not exist at "
                    f"{self.manifest}:{line}: {image_path}"
                )
            low_quality_value = row.get("low_quality_path", "").strip()
            low_quality_path = (
                Path(low_quality_value).expanduser()
                if low_quality_value
                else None
            )
            if low_quality_path is not None and not low_quality_path.is_absolute():
                low_quality_path = self.manifest.parent / low_quality_path
            if low_quality_path is not None and not low_quality_path.is_file():
                raise FileNotFoundError(
                    f"paired input from manifest does not exist at "
                    f"{self.manifest}:{line}: {low_quality_path}"
                )
            self.rows.append((image_path, task, low_quality_path))

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative; got {epoch}")
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> RestorationSample:
        if not 0 <= index < len(self.rows):
            raise IndexError(index)
        path, task, paired_input_path = self.rows[index]
        scale = TASK_SCALES[task]
        generator = torch.Generator(device="cpu").manual_seed(
            self.seed + self.epoch * len(self.rows) + index
        )
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            pixels = torch.frombuffer(
                bytearray(rgb.tobytes()), dtype=torch.uint8
            ).view(rgb.height, rgb.width, 3)
        clean = pixels.permute(2, 0, 1).contiguous().to(torch.float32).div_(255)
        paired_input = None
        if paired_input_path is not None:
            with Image.open(paired_input_path) as image:
                rgb = image.convert("RGB")
                input_pixels = torch.frombuffer(
                    bytearray(rgb.tobytes()), dtype=torch.uint8
                ).view(rgb.height, rgb.width, 3)
            paired_input = (
                input_pixels.permute(2, 0, 1).contiguous().to(torch.float32).div_(255)
            )

        target_size = self.patch_size
        height, width = clean.shape[-2:]
        if height < target_size or width < target_size:
            resize_scale = max(target_size / height, target_size / width)
            resized = F.interpolate(
                clean.unsqueeze(0),
                size=(
                    max(target_size, round(height * resize_scale)),
                    max(target_size, round(width * resize_scale)),
                ),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(0, 1)
            clean = resized
            height, width = clean.shape[-2:]
        if paired_input is not None and scale > 1:
            low_height, low_width = height // scale, width // scale
            if low_height == 0 or low_width == 0:
                raise ValueError(f"paired image is too small for {scale}x scale")
            paired_input = F.interpolate(
                paired_input.unsqueeze(0),
                size=(low_height, low_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(0, 1)
            top_units = (height - target_size) // scale
            left_units = (width - target_size) // scale
            top = int(torch.randint(top_units + 1, (), generator=generator)) * scale
            left = int(torch.randint(left_units + 1, (), generator=generator)) * scale
        else:
            top = int(torch.randint(height - target_size + 1, (), generator=generator))
            left = int(torch.randint(width - target_size + 1, (), generator=generator))
            if paired_input is not None and paired_input.shape[-2:] != (height, width):
                paired_input = F.interpolate(
                    paired_input.unsqueeze(0),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0).clamp(0, 1)
        clean = clean[:, top : top + target_size, left : left + target_size]

        if paired_input is not None:
            if scale > 1:
                low_top, low_left = top // scale, left // scale
                low_patch = target_size // scale
                low_quality = paired_input[
                    :,
                    low_top : low_top + low_patch,
                    low_left : low_left + low_patch,
                ]
            else:
                low_quality = paired_input[
                    :, top : top + target_size, left : left + target_size
                ]
        elif task.startswith("sr"):
            low_quality = F.interpolate(
                clean.unsqueeze(0),
                size=(target_size // scale, target_size // scale),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).squeeze(0).clamp(0, 1)
        elif task.startswith("denoise"):
            sigma = int(task.removeprefix("denoise"))
            noise = torch.randn(clean.shape, generator=generator)
            low_quality = (clean + noise * (sigma / 255.0)).clamp(0, 1)
        elif task == "derain":
            low_quality = (clean + 0.3 * _make_rain_layer(clean, generator)).clamp(0, 1)
        elif task == "desnow":
            flakes = (
                torch.rand((1, target_size, target_size), generator=generator) > 0.995
            ).to(clean.dtype)
            flakes = F.max_pool2d(flakes.unsqueeze(0), kernel_size=3, stride=1, padding=1)
            low_quality = clean * (1 - 0.75 * flakes[0]) + 0.75 * flakes[0]
        elif task == "dehaze":
            atmosphere = 0.8 + 0.2 * torch.rand((3, 1, 1), generator=generator)
            transmission = 0.65 + 0.25 * torch.rand((), generator=generator)
            low_quality = clean * transmission + atmosphere * (1 - transmission)
        elif task == "lowlight":
            gamma = 1.8 + 0.8 * float(torch.rand((), generator=generator))
            low_quality = clean.clamp_min(1e-6).pow(gamma)
            low_quality = (low_quality + torch.randn(clean.shape, generator=generator) * 0.01).clamp(0, 1)
        else:
            jpeg_pixels = (
                clean.mul(255)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
            )
            buffer = io.BytesIO()
            Image.frombytes(
                "RGB",
                (target_size, target_size),
                bytes(jpeg_pixels.untyped_storage()),
            ).save(buffer, format="JPEG", quality=30)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                jpeg = torch.frombuffer(
                    bytearray(compressed.convert("RGB").tobytes()),
                    dtype=torch.uint8,
                )
            low_quality = jpeg.view(target_size, target_size, 3).permute(
                2, 0, 1
            ).contiguous().to(torch.float32).div_(255)

        return RestorationSample(low_quality, clean, task, scale)
