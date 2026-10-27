"""Deterministic ten-image DIV2K loader with synthetic restoration corruptions."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image

Degradation = Literal["noise", "super_resolution", "rain"]
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class DIV2KMicroDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Use the first ``image_count`` sorted DIV2K images for paired training.

    Images are RGB float tensors in [0, 1]. Gaussian noise uses sigma 25 on an
    8-bit (0-255) scale. Super-resolution returns a bicubic 4x low-resolution
    input paired with its original high-resolution target. Rain streaks are
    blended into the high-resolution input so input and target shapes match.
    """

    def __init__(
        self,
        root: str | Path,
        degradation: Degradation,
        image_count: int = 10,
        seed: int = 0,
        noise_sigma_8bit: float = 25.0,
        scale_factor: int = 4,
        rain_strength: float = 0.3,
        patch_size: int | None = None,
        cache_noise: bool = False,
        cache_rain: bool = False,
    ) -> None:
        self.root = _resolve_image_directory(Path(root))
        self.degradation = degradation
        self.image_count = int(image_count)
        self.seed = int(seed)
        self.noise_sigma_8bit = float(noise_sigma_8bit)
        self.scale_factor = int(scale_factor)
        self.rain_strength = float(rain_strength)
        self.patch_size = None if patch_size is None else int(patch_size)
        self.cache_noise = bool(cache_noise and degradation == "noise")
        self.cache_rain = bool(cache_rain and degradation == "rain")
        self.epoch = 0
        self.fixed_crop = False

        if degradation not in ("noise", "super_resolution", "rain"):
            raise ValueError(
                "degradation must be 'noise', 'super_resolution', or 'rain'; "
                f"got {degradation!r}"
            )
        if self.image_count <= 0:
            raise ValueError(f"image_count must be positive; got {image_count}")
        if self.noise_sigma_8bit < 0:
            raise ValueError(
                f"noise_sigma_8bit must be non-negative; got {noise_sigma_8bit}"
            )
        if self.scale_factor < 2:
            raise ValueError(f"scale_factor must be at least 2; got {scale_factor}")
        if not 0.0 <= self.rain_strength <= 1.0:
            raise ValueError(
                f"rain_strength must be in [0, 1]; got {rain_strength}"
            )
        if self.patch_size is not None and self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive; got {patch_size}")

        images = sorted(
            path
            for path in self.root.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if len(images) < self.image_count:
            raise ValueError(
                f"expected at least {self.image_count} images in {self.root}, "
                f"found {len(images)}"
            )
        self.image_paths = images[: self.image_count]
        self._images = [_load_rgb_tensor(path) for path in self.image_paths]
        noise_cache_start = time.perf_counter()
        self._noisy_images = (
            [
                _make_noisy_image(
                    image,
                    self.noise_sigma_8bit,
                    torch.Generator(device="cpu").manual_seed(self.seed + index),
                )
                for index, image in enumerate(self._images)
            ]
            if self.cache_noise
            else None
        )
        self.noise_cache_initialization_ms = (
            (time.perf_counter() - noise_cache_start) * 1000
            if self._noisy_images is not None
            else 0.0
        )
        self.cached_noise_bytes = (
            sum(image.numel() * image.element_size() for image in self._noisy_images)
            if self._noisy_images is not None
            else 0
        )
        rain_cache_start = time.perf_counter()
        self._rainy_images = (
            [
                _make_rainy_image(
                    image,
                    self.rain_strength,
                    torch.Generator(device="cpu").manual_seed(self.seed + index),
                )
                for index, image in enumerate(self._images)
            ]
            if self.cache_rain
            else None
        )
        self.rain_cache_initialization_ms = (
            (time.perf_counter() - rain_cache_start) * 1000
            if self._rainy_images is not None
            else 0.0
        )
        self.cached_rain_bytes = (
            sum(image.numel() * image.element_size() for image in self._rainy_images)
            if self._rainy_images is not None
            else 0
        )

    def __len__(self) -> int:
        return self.image_count

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative; got {epoch}")
        self.epoch = int(epoch)

    def set_fixed_crop(self, fixed: bool = True) -> None:
        self.fixed_crop = bool(fixed)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < self.image_count:
            raise IndexError(index)
        clean = self._images[index]
        generator = torch.Generator(device="cpu").manual_seed(
            self.seed
            + (0 if self.fixed_crop else self.epoch * self.image_count)
            + index
        )
        if self.patch_size is not None:
            top, left = _random_crop_origin(clean, self.patch_size, generator)
            clean = clean[:, top : top + self.patch_size, left : left + self.patch_size]
            if self._noisy_images is not None:
                noisy = self._noisy_images[index][
                    :, top : top + self.patch_size, left : left + self.patch_size
                ]
            if self._rainy_images is not None:
                rainy = self._rainy_images[index][
                    :, top : top + self.patch_size, left : left + self.patch_size
                ]
        elif self._noisy_images is not None:
            noisy = self._noisy_images[index]
        elif self._rainy_images is not None:
            rainy = self._rainy_images[index]
        clean = clean.to(dtype=torch.float32).div_(255)

        if self.degradation == "noise":
            if self._noisy_images is not None:
                corrupted = noisy.to(dtype=torch.float32)
            else:
                noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
                corrupted = (
                    clean + noise * (self.noise_sigma_8bit / 255.0)
                ).clamp(0, 1)
        elif self.degradation == "super_resolution":
            height, width = clean.shape[-2:]
            low_height = height // self.scale_factor
            low_width = width // self.scale_factor
            if low_height == 0 or low_width == 0:
                raise ValueError(
                    f"image {self.image_paths[index]} is too small for "
                    f"{self.scale_factor}x downsampling"
                )
            corrupted = F.interpolate(
                clean.unsqueeze(0),
                size=(low_height, low_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).squeeze(0).clamp(0, 1)
        else:
            if self._rainy_images is not None:
                corrupted = rainy.to(dtype=torch.float32)
            else:
                corrupted = (
                    clean + self.rain_strength * _make_rain_layer(clean, generator)
                ).clamp(0, 1)

        return corrupted, clean


def create_micro_dataloader(
    root: str | Path,
    degradation: Degradation,
    *,
    image_count: int = 10,
    batch_size: int = 1,
    seed: int = 0,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    patch_size: int | None = None,
    cache_noise: bool = False,
    cache_rain: bool = False,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Construct a loader; batch size 1 supports variable image dimensions."""
    dataset = DIV2KMicroDataset(
        root=root,
        degradation=degradation,
        image_count=image_count,
        seed=seed,
        patch_size=patch_size,
        cache_noise=cache_noise,
        cache_rain=cache_rain,
    )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative; got {num_workers}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _resolve_image_directory(root: Path) -> Path:
    root = root.expanduser()
    candidates = (root, root / "DIV2K_train_HR")
    for candidate in candidates:
        if candidate.is_dir() and any(
            path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            for path in candidate.iterdir()
        ):
            return candidate
    expected = root / "DIV2K_train_HR"
    raise FileNotFoundError(
        f"no supported image files found in {root} or {expected}; "
        "pass the extracted DIV2K_train_HR directory or its parent"
    )


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
    return pixels.view(height, width, 3).permute(2, 0, 1).contiguous()


def _random_crop_origin(
    image: torch.Tensor,
    patch_size: int,
    generator: torch.Generator,
) -> tuple[int, int]:
    _, height, width = image.shape
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"image size {(height, width)} is smaller than patch_size={patch_size}"
        )
    top = int(torch.randint(height - patch_size + 1, (), generator=generator).item())
    left = int(torch.randint(width - patch_size + 1, (), generator=generator).item())
    return top, left


def _make_noisy_image(
    image: torch.Tensor,
    noise_sigma_8bit: float,
    generator: torch.Generator,
) -> torch.Tensor:
    clean = image.to(dtype=torch.float32).div_(255)
    noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    corrupted = (clean + noise * (noise_sigma_8bit / 255.0)).clamp_(0, 1)
    return corrupted.to(dtype=torch.float16)


def _make_rainy_image(
    image: torch.Tensor,
    rain_strength: float,
    generator: torch.Generator,
) -> torch.Tensor:
    clean = image.to(dtype=torch.float32).div_(255)
    corrupted = (clean + rain_strength * _make_rain_layer(clean, generator)).clamp_(0, 1)
    return corrupted.to(dtype=torch.float16)


def _make_rain_layer(
    image: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    _, height, width = image.shape
    grid_height = max(1, height // 8)
    grid_width = max(1, width // 8)
    impulses = (
        torch.rand((1, 1, grid_height, grid_width), generator=generator) > 0.985
    ).to(dtype=image.dtype)
    diagonal_kernel = image.new_tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0]]
    ).view(1, 1, 5, 5)
    streaks = F.conv2d(impulses, diagonal_kernel, padding=2)
    rain = F.interpolate(
        streaks, size=(height, width), mode="bilinear", align_corners=False
    ).squeeze(0)
    return rain.expand(3, -1, -1).clamp(0, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="DIV2K_train_HR directory or its parent directory",
    )
    parser.add_argument(
        "--degradation",
        choices=("noise", "super_resolution", "rain"),
        default="noise",
    )
    parser.add_argument("--image-count", type=int, default=10)
    args = parser.parse_args()

    loader = create_micro_dataloader(
        args.root,
        args.degradation,
        image_count=args.image_count,
    )
    corrupted, clean = next(iter(loader))
    print(f"images={len(loader.dataset)} degradation={args.degradation}")
    print(f"corrupted={tuple(corrupted.shape)} clean={tuple(clean.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
