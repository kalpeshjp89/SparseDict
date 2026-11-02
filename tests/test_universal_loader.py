import csv

import pytest
import torch
from PIL import Image

from data.universal_loader import TASK_SCALES, UniversalRestorationDataset


@pytest.fixture
def image_manifest(tmp_path):
    image_path = tmp_path / "source.png"
    image = Image.new("RGB", (96, 96))
    image.putdata(
        [
            (column * 2, row * 2, (column + row) % 256)
            for row in range(96)
            for column in range(96)
        ]
    )
    image.save(image_path)
    manifest = tmp_path / "samples.csv"
    with manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("image_path", "task", "low_quality_path"))
        for task in TASK_SCALES:
            writer.writerow((image_path, task, ""))
    return manifest


@pytest.mark.parametrize("task,scale", TASK_SCALES.items())
def test_universal_dataset_generates_finite_paired_patches(
    image_manifest, task, scale
):
    dataset = UniversalRestorationDataset(image_manifest, patch_size=36, seed=13)
    sample = dataset[list(TASK_SCALES).index(task)]

    assert sample.task == task
    assert sample.scale_factor == scale
    assert sample.clean.shape == (3, 36, 36)
    assert sample.low_quality.shape == (3, 36 // scale, 36 // scale)
    assert torch.isfinite(sample.low_quality).all()
    assert torch.isfinite(sample.clean).all()


def test_universal_dataset_is_reproducible_per_epoch(image_manifest):
    dataset = UniversalRestorationDataset(image_manifest, patch_size=36, seed=13)
    index = list(TASK_SCALES).index("denoise25")
    first = dataset[index]
    assert torch.equal(first.low_quality, dataset[index].low_quality)
    dataset.set_epoch(1)
    assert not torch.equal(first.low_quality, dataset[index].low_quality)


def test_universal_dataset_rejects_missing_image(image_manifest, tmp_path):
    manifest = tmp_path / "missing.csv"
    manifest.write_text(
        "image_path,task\nabsent.png,sr2\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        UniversalRestorationDataset(manifest, patch_size=32)


def test_universal_dataset_uses_paired_low_quality_sr_input(tmp_path):
    clean_path = tmp_path / "clean.png"
    low_quality_path = tmp_path / "low.png"
    Image.new("RGB", (64, 64), color=(200, 80, 20)).save(clean_path)
    Image.new("RGB", (32, 32), color=(100, 40, 10)).save(low_quality_path)
    manifest = tmp_path / "paired.csv"
    with manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("image_path", "task", "low_quality_path"))
        writer.writerow((clean_path, "sr2", low_quality_path))

    sample = UniversalRestorationDataset(manifest, patch_size=32)[0]

    assert sample.low_quality.shape == (3, 16, 16)
    assert sample.clean.shape == (3, 32, 32)
    expected_mean = torch.tensor([100 / 255, 40 / 255, 10 / 255])
    assert torch.allclose(
        sample.low_quality.mean(dim=(1, 2)), expected_mean, atol=0.01
    )
