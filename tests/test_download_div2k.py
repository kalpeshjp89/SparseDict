from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from scripts import download_div2k


def make_archive(path: Path, image_count: int = 12) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(1, image_count + 1):
            image_bytes = io.BytesIO()
            Image.new("RGB", (8, 6), (index, 40, 90)).save(image_bytes, format="PNG")
            archive.writestr(
                f"DIV2K_train_HR/{index:04d}.png", image_bytes.getvalue()
            )
        archive.writestr("unrelated/readme.txt", "not a training image")
    return path


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers=None):
        self._body = io.BytesIO(body)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._body.close()

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)


def test_extracts_requested_sorted_images_only(tmp_path):
    archive_path = make_archive(tmp_path / "source.zip")

    image_dir = download_div2k.extract_training_images(
        archive_path, tmp_path / "output", image_count=10
    )

    assert sorted(path.name for path in image_dir.glob("*.png")) == [
        f"{index:04d}.png" for index in range(1, 11)
    ]
    with Image.open(image_dir / "0001.png") as image:
        assert image.size == (8, 6)


def test_can_extract_full_archive_count(tmp_path):
    archive_path = make_archive(tmp_path / "source.zip", image_count=12)

    image_dir = download_div2k.extract_training_images(
        archive_path, tmp_path / "output", image_count=12
    )

    assert len(list(image_dir.glob("*.png"))) == 12


def test_resumes_partial_extraction_without_overwriting_existing_images(tmp_path):
    archive_path = make_archive(tmp_path / "source.zip")
    image_dir = tmp_path / "output" / "DIV2K_train_HR"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 6), (1, 40, 90)).save(image_dir / "0001.png")
    original = (image_dir / "0001.png").read_bytes()

    extracted = download_div2k.extract_training_images(
        archive_path, tmp_path / "output", image_count=10
    )

    assert len(list(extracted.glob("*.png"))) == 10
    assert (extracted / "0001.png").read_bytes() == original


def test_rejects_unsafe_archive_paths(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")
        archive.writestr("DIV2K_train_HR/0001.png", b"not an image")

    with pytest.raises(ValueError, match="unsafe path"):
        download_div2k.extract_training_images(
            archive_path, tmp_path / "output", image_count=1
        )


def test_download_archive_writes_and_validates_zip(tmp_path, monkeypatch):
    source_path = make_archive(tmp_path / "source.zip")
    archive_bytes = source_path.read_bytes()
    monkeypatch.setattr(
        download_div2k.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(
            archive_bytes, headers={"Content-Length": str(len(archive_bytes))}
        ),
    )

    downloaded = download_div2k.download_archive(
        "https://example.invalid/archive.zip", tmp_path / "download.zip"
    )

    assert downloaded.read_bytes() == archive_bytes
    assert not downloaded.with_suffix(".zip.part").exists()


def test_download_resumes_from_partial_file(tmp_path, monkeypatch):
    source_path = make_archive(tmp_path / "source.zip")
    archive_bytes = source_path.read_bytes()
    split = len(archive_bytes) // 2
    archive_path = tmp_path / "download.zip"
    partial_path = archive_path.with_suffix(".zip.part")
    partial_path.write_bytes(archive_bytes[:split])
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return FakeResponse(
            archive_bytes[split:],
            status=206,
            headers={"Content-Range": f"bytes {split}-{len(archive_bytes)-1}/{len(archive_bytes)}"},
        )

    monkeypatch.setattr(download_div2k.urllib.request, "urlopen", fake_urlopen)

    downloaded = download_div2k.download_archive(
        "https://example.invalid/archive.zip", archive_path
    )

    assert requests[0].get_header("Range") == f"bytes={split}-"
    assert downloaded.read_bytes() == archive_bytes


def test_cli_requires_license_acknowledgement(tmp_path):
    with pytest.raises(SystemExit) as error:
        download_div2k.main(["--output-dir", str(tmp_path)])

    assert error.value.code == 2


def test_cli_skips_download_when_requested_images_are_ready(tmp_path, monkeypatch):
    image_dir = tmp_path / "DIV2K_train_HR"
    image_dir.mkdir()
    for index in range(1, 11):
        Image.new("RGB", (8, 6), (index, 40, 90)).save(
            image_dir / f"{index:04d}.png"
        )
    monkeypatch.setattr(
        download_div2k,
        "download_archive",
        lambda *args, **kwargs: pytest.fail("download should be skipped"),
    )

    assert download_div2k.main(
        ["--output-dir", str(tmp_path), "--accept-license"]
    ) == 0
