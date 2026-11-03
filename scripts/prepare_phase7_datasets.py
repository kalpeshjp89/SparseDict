"""Acquire Phase 7 datasets from public sources or mounted Google Drive files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
from urllib.parse import urlparse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PUBLIC_DATASET_URLS = {
    # cv.snu.ac.kr (EDSR host) is intermittently dead; the Flickr2K HR-only
    # zip on Hugging Face mirrors the same 2650 HR images (~10.85 GB) plus
    # LR folders and is sufficient because degradations are synthesized
    # from clean images.
    "flickr2k": (
        "https://huggingface.co/datasets/yangtao9009/Flickr2K/"
        "resolve/main/Flickr2K.zip"
    ),
    "cbsd68": (
        "https://github.com/clausmichele/CBSD68-dataset/"
        "archive/refs/heads/master.zip"
    ),
    # Original SNU-hosted Urban100.tar is unreachable; the Hugging Face
    # raw copy holds the 100 HR images (129 MB) used for benchmarking.
    "urban100": (
        "https://huggingface.co/datasets/eugenesiow/Urban100/"
        "resolve/main/data/Urban100_HR.tar.gz"
    ),
}
# Source-URL schemes accepted in --source-url in addition to direct
# HTTPS archive links:
#   kaggle:<owner>/<slug>     -> `kaggle datasets download` (requires the
#                                user's kaggle.json credentials on the host)
#   hf-dataset:<repo_id>      -> huggingface_hub.snapshot_download for
#                                spread (per-image) dataset repositories
KAGGLE_SCHEME = "kaggle:"
HF_DATASET_SCHEME = "hf-dataset:"
DATASET_NAMES = {
    "div2k": "DIV2K",
    "flickr2k": "Flickr2K",
    "cbsd68": "CBSD68",
    "sots": "SOTS",
    "outdoor-rain": "Outdoor-Rain",
    "urban100": "Urban100",
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CHUNK_SIZE = 8 * 1024 * 1024


def _parse_mapping(values: list[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        key = key.strip().lower()
        item = item.strip()
        if not separator or key not in DATASET_NAMES or not item:
            raise ValueError(f"{option} expects DATASET=value; got {value!r}")
        result[key] = item
    return result


def _contains_images(path: Path) -> bool:
    return path.is_dir() and any(
        candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
        for candidate in path.rglob("*")
    )


def _download_resumable(url: str, archive: Path) -> Path:
    if urlparse(url).scheme.lower() != "https":
        raise ValueError(f"Dataset source URLs must use HTTPS: {url}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not download {url}: {error}") from error

    with response:
        if offset and response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise RuntimeError(
                    f"Invalid range response while resuming {url}: {content_range!r}"
                )
            mode = "ab"
        elif response.status == 200:
            offset = 0
            mode = "wb"
        else:
            raise RuntimeError(f"Unexpected HTTP status {response.status} for {url}")
        with partial.open(mode) as output:
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
    partial.replace(archive)
    return archive


def _download_google_drive(file_id: str, archive: Path) -> Path:
    try:
        import gdown
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gdown"],
            check=True,
        )
        import gdown

    archive.parent.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download(id=file_id, output=str(archive), quiet=False)
    if not downloaded or not archive.is_file() or archive.stat().st_size == 0:
        raise RuntimeError(
            "Google Drive did not return an archive. Check the file ID and ensure "
            "the file is accessible to this Colab account."
        )
    return archive


def _download_kaggle(slug: str, destination: Path) -> None:
    """Download a Kaggle dataset archive with the Kaggle CLI.

    ``kaggle`` credentials must already be configured by the user on the
    host (``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME``/
    ``KAGGLE_KEY`` environment variables); this helper never handles the
    credentials itself.
    """
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        slug,
        "-p",
        str(destination),
        "--unzip",
        "-o",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True)
    except FileNotFoundError:
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"kaggle datasets download failed for {slug!r}: "
            f"{detail or completed.returncode}"
        )


def _download_hf_dataset(repo_id: str, destination: Path) -> None:
    """Download a spread Hugging Face dataset repository in-place."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "huggingface_hub"],
            check=True,
        )
        from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(destination),
    )


def _safe_destination(root: Path, member_name: str) -> Path:
    member = PurePosixPath(member_name)
    windows_member = PureWindowsPath(member_name)
    if (
        "\\" in member_name
        or member.is_absolute()
        or windows_member.is_absolute()
        or windows_member.drive
        or ".." in member.parts
    ):
        raise ValueError(f"Archive contains an unsafe path: {member_name!r}")
    destination = root.joinpath(*member.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Archive path escapes its extraction directory: {member_name!r}")
    return destination


def _extract_archive(archive: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Extraction destination is not empty: {destination}. "
            "Choose a separate dataset root or move the existing files first."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-extract-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                for item in source.infolist():
                    target = _safe_destination(staging, item.filename)
                    if item.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(item) as input_stream, target.open("wb") as output:
                        shutil.copyfileobj(input_stream, output, length=CHUNK_SIZE)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive, "r:*") as source:
                for item in source.getmembers():
                    target = _safe_destination(staging, item.name)
                    if item.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif item.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        input_stream = source.extractfile(item)
                        if input_stream is None:
                            raise ValueError(f"Could not read archive member {item.name}")
                        with input_stream, target.open("wb") as output:
                            shutil.copyfileobj(input_stream, output, length=CHUNK_SIZE)
                    else:
                        raise ValueError(
                            f"Archive contains unsupported link/device: {item.name!r}"
                        )
        else:
            raise ValueError(
                f"Unsupported archive format for {archive}; expected ZIP or TAR."
            )
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)


def _resolve_clean_image_dir(dataset: str, dataset_root: Path) -> Path:
    if not _contains_images(dataset_root):
        raise FileNotFoundError(f"No images found under {dataset_root}")

    preferred = {
        "cbsd68": ("original_png", "original", "clean"),
        "sots": ("clear", "gt", "ground_truth", "clean", "target"),
        "outdoor-rain": ("clean", "gt", "ground_truth", "target"),
        "flickr2k": ("flickr2k",),
        "urban100": ("urban100",),
    }.get(dataset, ())
    candidates = [
        path
        for path in dataset_root.rglob("*")
        if path.is_dir()
        and path.name.casefold() in {name.casefold() for name in preferred}
        and _contains_images(path)
    ]
    if candidates:
        return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]
    if dataset in {"sots", "outdoor-rain"}:
        raise FileNotFoundError(
            f"Found images under {dataset_root}, but no clean/ground-truth image "
            "folder. Set the notebook's DATASET_DIRS entry to the dataset's "
            "clear/clean/GT directory; do not train on hazy/rainy inputs as targets."
        )
    return dataset_root


def resolve_clean_image_dir(dataset: str, root: str | Path) -> Path:
    """Return a dataset's clean-image directory, preferring known folder names."""
    key = dataset.strip().lower()
    if key not in DATASET_NAMES:
        raise ValueError(f"Unsupported Phase 7 dataset: {dataset}")
    dataset_root = Path(root) / DATASET_NAMES[key]
    if key == "div2k":
        candidates = (
            Path(root) / "DIV2K_train_HR",
            Path(root) / "DIV2K" / "DIV2K_train_HR",
        )
        dataset_root = next(
            (path for path in candidates if _contains_images(path)), candidates[0]
        )
    return _resolve_clean_image_dir(key, dataset_root)


def prepare_dataset(
    key: str,
    root: Path,
    download_public: bool,
    accept_div2k_license: bool,
    source_urls: dict[str, str],
    drive_file_ids: dict[str, str],
    drive_archives: dict[str, str],
) -> Path:
    name = DATASET_NAMES[key]
    if key == "div2k":
        image_dir = root / "DIV2K_train_HR"
        legacy_image_dir = root / "DIV2K" / "DIV2K_train_HR"
        if _contains_images(image_dir):
            print(f"{name}: using existing Drive images at {image_dir}")
            return image_dir
        if _contains_images(legacy_image_dir):
            print(f"{name}: using existing Drive images at {legacy_image_dir}")
            return legacy_image_dir
        if not accept_div2k_license:
            raise PermissionError(
                "DIV2K is restricted to academic research use. Review its terms, "
                "then set ACCEPT_DIV2K_TERMS=True in the notebook to download."
            )
        archive_path = root / "DIV2K_train_HR.zip"
        if key in drive_archives:
            drive_archive = Path(drive_archives[key]).expanduser()
            if not drive_archive.is_file():
                raise FileNotFoundError(f"Drive archive not found: {drive_archive}")
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(drive_archive, archive_path)
        elif key in drive_file_ids:
            _download_google_drive(drive_file_ids[key], archive_path)
        elif key in source_urls:
            _download_resumable(source_urls[key], archive_path)
        elif not download_public:
            raise FileNotFoundError(
                f"DIV2K is missing at {image_dir}. Enable DOWNLOAD_PUBLIC_DATASETS "
                "or place the archive/images under the Drive dataset root."
            )
        from scripts.download_div2k import main as download_div2k_main

        download_div2k_main(
            [
                "--output-dir",
                str(root),
                "--image-count",
                "800",
                "--accept-license",
            ]
        )
        return image_dir

    dataset_root = root / name
    if _contains_images(dataset_root):
        print(f"{name}: using existing Drive images at {dataset_root}")
        return _resolve_clean_image_dir(key, dataset_root)

    archive = root / "_archives" / name
    if key in drive_archives:
        drive_archive = Path(drive_archives[key]).expanduser()
        if not drive_archive.is_file():
            raise FileNotFoundError(f"Drive archive not found: {drive_archive}")
        archive = root / "_archives" / drive_archive.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        if drive_archive.resolve() != archive.resolve():
            shutil.copy2(drive_archive, archive)
    elif key in drive_file_ids:
        archive = root / "_archives" / f"{name}.download"
        _download_google_drive(drive_file_ids[key], archive)
    elif key in source_urls:
        source = source_urls[key]
        if source.startswith(KAGGLE_SCHEME):
            _download_kaggle(source[len(KAGGLE_SCHEME) :], dataset_root)
            return _resolve_clean_image_dir(key, dataset_root)
        if source.startswith(HF_DATASET_SCHEME):
            _download_hf_dataset(
                source[len(HF_DATASET_SCHEME) :], dataset_root
            )
            return _resolve_clean_image_dir(key, dataset_root)
        archive = _download_resumable(source, archive.with_suffix(
            Path(source.split("?", maxsplit=1)[0]).suffix or ".archive"
        ))
    elif download_public and key in PUBLIC_DATASET_URLS:
        url = PUBLIC_DATASET_URLS[key]
        archive = _download_resumable(
            url,
            archive.with_suffix(Path(url.split("?", maxsplit=1)[0]).suffix or ".archive"),
        )
    else:
        source = PUBLIC_DATASET_URLS.get(key, "no verified direct-download URL")
        raise FileNotFoundError(
            f"{name} is missing under {dataset_root}. Copy its extracted folder "
            "there, set a mounted Drive archive path/file ID, or provide an "
            f"authorized direct HTTPS URL. Public source, when available: {source}"
        )

    _extract_archive(archive, dataset_root)
    return _resolve_clean_image_dir(key, dataset_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(DATASET_NAMES),
        required=True,
        help="Dataset to prepare; repeat for multiple datasets.",
    )
    parser.add_argument(
        "--download-public",
        action="store_true",
        help="Download datasets with a built-in public source URL.",
    )
    parser.add_argument(
        "--accept-div2k-license",
        action="store_true",
        help="Confirm DIV2K academic-research terms and copyright restrictions.",
    )
    parser.add_argument(
        "--source-url",
        action="append",
        default=[],
        metavar="DATASET=URL",
        help=(
            "Authorized direct HTTPS archive URL, `kaggle:owner/slug`, or "
            "`hf-dataset:repo_id` for a dataset."
        ),
    )
    parser.add_argument(
        "--drive-file-id",
        action="append",
        default=[],
        metavar="DATASET=ID",
        help="Google Drive file ID for a shared dataset archive.",
    )
    parser.add_argument(
        "--drive-archive",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Path to a dataset archive already present on mounted Drive.",
    )
    args = parser.parse_args(argv)
    try:
        source_urls = _parse_mapping(args.source_url, "--source-url")
        drive_file_ids = _parse_mapping(args.drive_file_id, "--drive-file-id")
        drive_archives = _parse_mapping(args.drive_archive, "--drive-archive")
        root = args.root.expanduser()
        root.mkdir(parents=True, exist_ok=True)
        for key in args.dataset:
            image_dir = prepare_dataset(
                key,
                root,
                args.download_public,
                args.accept_div2k_license,
                source_urls,
                drive_file_ids,
                drive_archives,
            )
            print(f"{DATASET_NAMES[key]}_IMAGE_DIR={image_dir}")
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
