import zipfile

import pytest

from scripts.prepare_phase7_datasets import (
    _extract_archive,
    prepare_dataset,
    resolve_clean_image_dir,
)


def test_existing_drive_dataset_is_reused_without_downloading(tmp_path):
    images = tmp_path / "Flickr2K" / "Flickr2K"
    images.mkdir(parents=True)
    (images / "sample.png").write_bytes(b"placeholder")

    result = prepare_dataset(
        "flickr2k",
        tmp_path,
        download_public=False,
        accept_div2k_license=False,
        source_urls={},
        drive_file_ids={},
        drive_archives={},
    )

    assert result == images


def test_resolve_prefers_clean_sots_folder(tmp_path):
    dataset_root = tmp_path / "SOTS"
    (dataset_root / "hazy").mkdir(parents=True)
    (dataset_root / "clear").mkdir()
    (dataset_root / "hazy" / "sample.png").write_bytes(b"hazy")
    (dataset_root / "clear" / "sample.png").write_bytes(b"clear")

    assert resolve_clean_image_dir("sots", tmp_path) == dataset_root / "clear"


def test_archive_extracts_files_into_dataset_folder(tmp_path):
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("source/clean/image.png", b"image")

    destination = tmp_path / "dataset"
    _extract_archive(archive_path, destination)

    assert (destination / "source" / "clean" / "image.png").read_bytes() == b"image"


def test_drive_archive_is_extracted_and_clean_folder_selected(tmp_path):
    drive_archive = tmp_path / "drive" / "CBSD68.zip"
    drive_archive.parent.mkdir()
    with zipfile.ZipFile(drive_archive, "w") as archive:
        archive.writestr(
            "CBSD68-dataset-master/CBSD68/original_png/image.png",
            b"clean image",
        )
        archive.writestr(
            "CBSD68-dataset-master/CBSD68/noisy25/image.png",
            b"noisy image",
        )

    image_dir = prepare_dataset(
        "cbsd68",
        tmp_path / "datasets",
        download_public=True,
        accept_div2k_license=False,
        source_urls={},
        drive_file_ids={},
        drive_archives={"cbsd68": str(drive_archive)},
    )

    assert image_dir == (
        tmp_path
        / "datasets"
        / "CBSD68"
        / "CBSD68-dataset-master"
        / "CBSD68"
        / "original_png"
    )


def test_archive_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", b"unsafe")

    with pytest.raises(ValueError, match="unsafe path"):
        _extract_archive(archive_path, tmp_path / "dataset")
    assert not (tmp_path / "outside.txt").exists()


def test_archive_rejects_windows_drive_paths(tmp_path):
    archive_path = tmp_path / "windows-path.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("C:/outside.txt", b"unsafe")

    with pytest.raises(ValueError, match="unsafe path"):
        _extract_archive(archive_path, tmp_path / "dataset")


def test_missing_restricted_dataset_reports_drive_or_url_options(tmp_path):
    with pytest.raises(FileNotFoundError, match="Drive archive"):
        prepare_dataset(
            "sots",
            tmp_path,
            download_public=True,
            accept_div2k_license=False,
            source_urls={},
            drive_file_ids={},
            drive_archives={},
        )


def test_kaggle_scheme_source_downloads_into_dataset_root(monkeypatch, tmp_path):
    from scripts import prepare_phase7_datasets as module

    calls = {}

    def fake_download_kaggle(slug, destination):
        calls["slug"] = slug
        clean = destination / "clear"
        clean.mkdir(parents=True, exist_ok=True)
        (clean / "sample.png").write_bytes(b"clean")

    monkeypatch.setattr(module, "_download_kaggle", fake_download_kaggle)

    result = prepare_dataset(
        "sots",
        tmp_path,
        download_public=False,
        accept_div2k_license=False,
        source_urls={
            "sots": "kaggle:balraj98/synthetic-objective-testing-set-sots-reside"
        },
        drive_file_ids={},
        drive_archives={},
    )

    assert calls["slug"] == "balraj98/synthetic-objective-testing-set-sots-reside"
    assert result == tmp_path / "SOTS" / "clear"


def test_hf_dataset_scheme_source_downloads_into_dataset_root(monkeypatch, tmp_path):
    from scripts import prepare_phase7_datasets as module

    calls = {}

    def fake_download_hf_dataset(repo_id, destination):
        calls["repo_id"] = repo_id
        target = destination / "target"
        target.mkdir(parents=True, exist_ok=True)
        (target / "sample.png").write_bytes(b"target")
        (destination / "input").mkdir()
        (destination / "input" / "sample.png").write_bytes(b"rainy")

    monkeypatch.setattr(module, "_download_hf_dataset", fake_download_hf_dataset)

    result = prepare_dataset(
        "outdoor-rain",
        tmp_path,
        download_public=False,
        accept_div2k_license=False,
        source_urls={"outdoor-rain": "hf-dataset:dronefreak/Rain13K"},
        drive_file_ids={},
        drive_archives={},
    )

    assert calls["repo_id"] == "dronefreak/Rain13K"
    assert result == tmp_path / "Outdoor-Rain" / "target"


def test_public_urls_no_longer_reference_dead_snu_host():
    from scripts.prepare_phase7_datasets import PUBLIC_DATASET_URLS

    for name, url in PUBLIC_DATASET_URLS.items():
        assert not url.startswith("http://cv.snu.ac.kr"), name
        assert not url.startswith("https://cv.snu.ac.kr"), name
        assert url.startswith("https://"), name
