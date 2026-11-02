"""Create reproducible train/validation CSV manifests from clean image folders."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.micro_loader import SUPPORTED_EXTENSIONS
from data.universal_loader import TASK_SCALES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        action="append",
        required=True,
        help="Clean image directory; repeat for additional dataset directories.",
    )
    parser.add_argument(
        "--tasks",
        default="sr2,sr3,sr4,denoise15,denoise25,denoise50,derain,desnow,dehaze,lowlight,jpeg",
        help="Comma-separated task names from data.universal_loader.TASK_SCALES.",
    )
    parser.add_argument(
        "--paired-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional CSV with image_path,low_quality_path,task columns; "
            "repeat to add real paired examples."
        ),
    )
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args(argv)

    tasks = [task.strip().lower() for task in args.tasks.split(",") if task.strip()]
    if not tasks or any(task not in TASK_SCALES for task in tasks):
        parser.error(f"--tasks must use names from {', '.join(TASK_SCALES)}")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")

    images: set[Path] = set()
    for directory in args.image_dir:
        if not directory.is_dir():
            parser.error(f"image directory does not exist: {directory}")
        images.update(
            path.resolve()
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    paired_rows: dict[Path, list[tuple[str, Path]]] = defaultdict(list)
    for paired_manifest in args.paired_manifest:
        if not paired_manifest.is_file():
            parser.error(f"paired manifest does not exist: {paired_manifest}")
        with paired_manifest.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = {"image_path", "low_quality_path", "task"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                parser.error(
                    f"{paired_manifest} must contain columns "
                    "image_path,low_quality_path,task"
                )
            for line, row in enumerate(reader, start=2):
                target = _resolve_path(paired_manifest, row["image_path"])
                low_quality = _resolve_path(
                    paired_manifest, row["low_quality_path"]
                )
                task = row["task"].strip().lower()
                if task not in TASK_SCALES:
                    parser.error(
                        f"unsupported paired task {task!r} at "
                        f"{paired_manifest}:{line}"
                    )
                if not target.is_file() or not low_quality.is_file():
                    parser.error(
                        f"paired paths must exist at {paired_manifest}:{line}"
                    )
                images.add(target)
                paired_rows[target].append((task, low_quality))
    images = sorted(images)
    if len(images) < 2:
        parser.error("at least two readable image files are required")

    random.Random(args.seed).shuffle(images)
    validation_count = max(1, round(len(images) * args.validation_fraction))
    validation_images = images[:validation_count]
    training_images = images[validation_count:]
    if not training_images:
        parser.error("the split left no training images")

    _write_manifest(args.train_output, training_images, tasks, paired_rows)
    _write_manifest(args.validation_output, validation_images, tasks, paired_rows)
    print(
        f"images={len(images)} train_images={len(training_images)} "
        f"validation_images={len(validation_images)} tasks={len(tasks)}"
    )
    print(f"train_manifest={args.train_output}")
    print(f"validation_manifest={args.validation_output}")
    return 0


def _resolve_path(manifest: Path, value: str) -> Path:
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def _write_manifest(
    path: Path,
    images: list[Path],
    tasks: list[str],
    paired_rows: dict[Path, list[tuple[str, Path]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("image_path", "task", "low_quality_path"))
        for image in images:
            for task in tasks:
                writer.writerow((str(image), task, ""))
            for task, low_quality in paired_rows.get(image, []):
                writer.writerow((str(image), task, str(low_quality)))


if __name__ == "__main__":
    raise SystemExit(main())
