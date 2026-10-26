"""Download the official DIV2K training-HR archive and extract a micro subset."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

DIV2K_TRAIN_HR_URL = (
    "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
)
ARCHIVE_NAME = "DIV2K_train_HR.zip"
IMAGE_DIRECTORY = "DIV2K_train_HR"
CHUNK_SIZE = 8 * 1024 * 1024
IMAGE_NAME_PATTERN = re.compile(r"^\d{4}\.png$")


def download_archive(url: str, archive_path: Path) -> Path:
    """Download with HTTP Range resume support, preserving partial files."""
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    offset = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)

    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.URLError as error:
        raise RuntimeError(f"failed to download DIV2K archive: {error}") from error

    with response:
        status = response.status
        if offset and status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
            if match is None or int(match.group(1)) != offset:
                raise RuntimeError(
                    "server returned an invalid Content-Range while resuming "
                    f"the download: {content_range!r}"
                )
            total_size = None if match.group(3) == "*" else int(match.group(3))
            mode = "ab"
        elif status == 200:
            offset = 0
            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) if content_length else None
            mode = "wb"
        else:
            raise RuntimeError(
                f"unexpected HTTP status {status} while downloading DIV2K"
            )

        downloaded = offset
        print(
            f"Downloading {url} ({_format_size(total_size) if total_size else 'size unknown'})"
        )
        with partial_path.open(mode) as output:
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    print(
                        f"\rDownloaded {_format_size(downloaded)} / "
                        f"{_format_size(total_size)} ({downloaded / total_size:.1%})",
                        end="",
                        flush=True,
                    )
        if total_size is not None and downloaded != total_size:
            print()
            raise RuntimeError(
                f"incomplete download: received {downloaded} of {total_size} bytes; "
                f"rerun to resume from {partial_path}"
            )
        print(f"\rDownloaded {_format_size(downloaded)}")

    try:
        with zipfile.ZipFile(partial_path) as archive:
            archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded file is not a valid ZIP archive: {error}") from error

    partial_path.replace(archive_path)
    return archive_path


def extract_training_images(
    archive_path: Path,
    output_dir: Path,
    image_count: int = 10,
) -> Path:
    """Extract and verify the first sorted training-HR PNGs only."""
    if image_count <= 0:
        raise ValueError(f"image_count must be positive; got {image_count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / IMAGE_DIRECTORY

    with zipfile.ZipFile(archive_path) as archive:
        image_infos = _training_image_infos(archive)
        if len(image_infos) < image_count:
            raise ValueError(
                f"archive contains {len(image_infos)} training images; "
                f"requested {image_count}"
            )
        selected = image_infos[:image_count]
        targets = [image_dir / PurePosixPath(info.filename).name for info in selected]
        pending = []
        for info, target in zip(selected, targets):
            if target.exists():
                with Image.open(target) as image:
                    image.verify()
            else:
                pending.append((info, target))
        if not pending:
            print(f"Found requested images already extracted in {image_dir}")
            return image_dir

        with tempfile.TemporaryDirectory(
            prefix=".div2k-extract-", dir=output_dir
        ) as temporary:
            staging_dir = Path(temporary)
            staged_paths: list[Path] = []
            for info, target in pending:
                staged_path = staging_dir / target.name
                with archive.open(info) as source, staged_path.open("wb") as output:
                    shutil.copyfileobj(source, output, length=CHUNK_SIZE)
                with Image.open(staged_path) as image:
                    image.verify()
                staged_paths.append(staged_path)

            image_dir.mkdir(parents=True, exist_ok=True)
            for staged_path, (_, target) in zip(staged_paths, pending):
                staged_path.replace(target)

    print(f"Extracted {len(pending)} training images to {image_dir}")
    return image_dir


def _training_image_infos(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    expected_prefix = f"{IMAGE_DIRECTORY}/"
    image_infos = []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"archive contains an unsafe path: {info.filename!r}")
        if info.is_dir() or not info.filename.startswith(expected_prefix):
            continue
        if len(path.parts) != 2 or not IMAGE_NAME_PATTERN.fullmatch(path.name):
            raise ValueError(
                f"unexpected file in {IMAGE_DIRECTORY} archive folder: {info.filename!r}"
            )
        image_infos.append(info)
    image_infos.sort(key=lambda info: info.filename)
    return image_infos


def _format_size(size: int | None) -> str:
    if size is None:
        return "unknown size"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _existing_subset_is_ready(image_dir: Path, image_count: int) -> bool:
    expected_paths = [
        image_dir / f"{index:04d}.png" for index in range(1, image_count + 1)
    ]
    if not all(path.is_file() for path in expected_paths):
        return False
    for path in expected_paths:
        with Image.open(path) as image:
            image.verify()
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/content/DIV2K"),
        help="download and extraction directory (default: /content/DIV2K)",
    )
    parser.add_argument(
        "--image-count",
        type=int,
        default=10,
        help="number of sorted training-HR images to extract (default: 10; max: 800)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="retain the 3.28 GiB ZIP after successful extraction",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="confirm use is for academic research and respect original image copyrights",
    )
    args = parser.parse_args(argv)

    if not args.accept_license:
        parser.error(
            "DIV2K is made available for academic research only and image "
            "copyright belongs to the original owners; pass --accept-license "
            "only if this use complies"
        )
    if not 1 <= args.image_count <= 800:
        parser.error("--image-count must be between 1 and 800")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / IMAGE_DIRECTORY
    if _existing_subset_is_ready(image_dir, args.image_count):
        print(f"Found requested images already extracted in {image_dir}")
        print(f"Dataset ready: {image_dir}")
        print(f"Loader root: {output_dir}")
        return 0

    archive_path = output_dir / ARCHIVE_NAME
    if archive_path.exists():
        print(f"Using existing archive {archive_path}")
    else:
        download_archive(DIV2K_TRAIN_HR_URL, archive_path)

    image_dir = extract_training_images(
        archive_path, output_dir, image_count=args.image_count
    )
    if not args.keep_archive:
        archive_path.unlink()
        print(f"Removed downloaded archive to free space: {archive_path}")

    print(f"Dataset ready: {image_dir}")
    print(f"Loader root: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
