import csv

from PIL import Image

from scripts.make_universal_manifest import main


def test_manifest_builder_splits_images_and_keeps_real_pairs(tmp_path):
    image_dir = tmp_path / "clean"
    image_dir.mkdir()
    targets = []
    for index in range(2):
        target = image_dir / f"clean{index}.png"
        Image.new("RGB", (32, 32), color=(index * 40, 80, 120)).save(target)
        targets.append(target)
    paired_input = tmp_path / "rainy.png"
    Image.new("RGB", (32, 32), color=(120, 80, 40)).save(paired_input)
    paired_csv = tmp_path / "paired.csv"
    with paired_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("image_path", "low_quality_path", "task"))
        writer.writerow((targets[0], paired_input, "derain"))
    train_csv = tmp_path / "train.csv"
    validation_csv = tmp_path / "validation.csv"

    assert (
        main(
            [
                "--image-dir",
                str(image_dir),
                "--tasks",
                "denoise25",
                "--paired-manifest",
                str(paired_csv),
                "--train-output",
                str(train_csv),
                "--validation-output",
                str(validation_csv),
                "--validation-fraction",
                "0.5",
                "--seed",
                "1",
            ]
        )
        == 0
    )

    records = []
    for path in (train_csv, validation_csv):
        with path.open(newline="", encoding="utf-8") as source:
            records.extend(csv.DictReader(source))
    assert len(records) == 3
    assert any(
        row["task"] == "derain"
        and row["image_path"] == str(targets[0].resolve())
        and row["low_quality_path"] == str(paired_input.resolve())
        for row in records
    )
