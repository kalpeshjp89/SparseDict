from pathlib import Path

import pytest
import torch
from PIL import Image

from data.micro_loader import DIV2KMicroDataset, create_micro_dataloader


@pytest.fixture
def image_directory(tmp_path: Path) -> Path:
    image_dir = tmp_path / "DIV2K_train_HR"
    image_dir.mkdir()
    for index in range(12):
        width = 32 + index % 2 * 4
        height = 40 + index % 2 * 4
        image = Image.new("RGB", (width, height))
        image.putdata(
            [
                ((x * 7 + index) % 256, (y * 5 + index) % 256, (x + y) % 256)
                for y in range(height)
                for x in range(width)
            ]
        )
        image.save(image_dir / f"{index:04d}.png")
    return image_dir


@pytest.mark.parametrize("degradation", ["noise", "super_resolution", "rain"])
def test_dataset_loads_exactly_ten_paired_images(image_directory, degradation):
    dataset = DIV2KMicroDataset(image_directory, degradation=degradation)
    corrupted, clean = dataset[0]

    assert len(dataset) == 10
    assert clean.shape[0] == 3
    assert clean.dtype == torch.float32
    assert torch.all((clean >= 0) & (clean <= 1))
    if degradation == "super_resolution":
        assert corrupted.shape[-2:] == (clean.shape[-2] // 4, clean.shape[-1] // 4)
    else:
        assert corrupted.shape == clean.shape
    assert torch.all((corrupted >= 0) & (corrupted <= 1))


def test_noise_is_deterministic_for_same_seed_and_image(image_directory):
    first = DIV2KMicroDataset(image_directory, "noise", seed=17)
    second = DIV2KMicroDataset(image_directory, "noise", seed=17)

    assert torch.equal(first[0][0], second[0][0])
    assert not torch.equal(first[0][0], first[1][0])


def test_rain_is_deterministic_and_changes_image(image_directory):
    first = DIV2KMicroDataset(image_directory, "rain", seed=4)
    second = DIV2KMicroDataset(image_directory, "rain", seed=4)

    rain, clean = first[0]
    assert torch.equal(rain, second[0][0])
    assert torch.any(rain > clean)


def test_loader_yields_corrupted_clean_pairs(image_directory):
    loader = create_micro_dataloader(
        image_directory, "noise", image_count=10, batch_size=1
    )

    corrupted, clean = next(iter(loader))
    assert len(loader.dataset) == 10
    assert corrupted.shape == clean.shape == (1, 3, 40, 32)


def test_dataset_uses_cached_images_after_source_files_are_removed(image_directory):
    dataset = DIV2KMicroDataset(image_directory, "noise", image_count=1)
    expected_clean = dataset[0][1]
    dataset.image_paths[0].unlink()

    corrupted, clean = dataset[0]

    assert torch.equal(clean, expected_clean)
    assert corrupted.shape == clean.shape


def test_accepts_parent_directory_containing_div2k_folder(image_directory):
    dataset = DIV2KMicroDataset(image_directory.parent, "rain")

    assert len(dataset) == 10


def test_random_patch_crops_vary_by_epoch_and_fixed_crop_is_reproducible(image_directory):
    dataset = DIV2KMicroDataset(
        image_directory, "noise", image_count=1, patch_size=16, seed=7
    )
    first_pair = dataset[0]
    dataset.set_epoch(1)
    second_pair = dataset[0]
    assert first_pair[0].shape == first_pair[1].shape == (3, 16, 16)
    assert not torch.equal(first_pair[1], second_pair[1])

    dataset.set_fixed_crop()
    fixed_pair = dataset[0]
    dataset.set_epoch(5)
    assert torch.equal(fixed_pair[1], dataset[0][1])


def test_rejects_too_few_images_and_invalid_degradations(tmp_path, image_directory):
    with pytest.raises(ValueError, match="expected at least 13 images"):
        DIV2KMicroDataset(image_directory, "noise", image_count=13)
    with pytest.raises(ValueError, match="degradation must be"):
        DIV2KMicroDataset(image_directory, "unknown")
    with pytest.raises(ValueError, match="image_count must be positive"):
        DIV2KMicroDataset(image_directory, "noise", image_count=0)
    with pytest.raises(ValueError, match="patch_size must be positive"):
        DIV2KMicroDataset(image_directory, "noise", patch_size=0)
    with pytest.raises(ValueError, match="smaller than patch_size"):
        DIV2KMicroDataset(
            image_directory, "noise", image_count=1, patch_size=100
        )[0]
    with pytest.raises(FileNotFoundError, match="no supported image files"):
        DIV2KMicroDataset(tmp_path / "missing", "noise")
