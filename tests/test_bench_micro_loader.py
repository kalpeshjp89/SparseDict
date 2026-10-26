import json
from pathlib import Path

from PIL import Image

from experiments.bench_micro_loader import main as benchmark_main


def make_images(directory: Path) -> Path:
    image_directory = directory / "DIV2K_train_HR"
    image_directory.mkdir()
    for index in range(2):
        Image.new("RGB", (32, 32), color=(80 + index, 120, 160)).save(
            image_directory / f"{index + 1:04d}.png"
        )
    return image_directory


def test_benchmark_reports_all_modes_and_writes_json(tmp_path, capsys):
    image_directory = make_images(tmp_path)
    output_path = tmp_path / "metrics" / "loader.json"

    result = benchmark_main(
        [
            "--data-root",
            str(image_directory),
            "--image-count",
            "2",
            "--warmup-passes",
            "0",
            "--repeats",
            "1",
            "--max-ms",
            "100000",
            "--json-output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 0
    assert report["pass"] is True
    assert [entry["degradation"] for entry in report["results"]] == [
        "noise",
        "super_resolution",
        "rain",
    ]
    assert all(entry["samples"] == 2 for entry in report["results"])
    assert "overall pass=True" in capsys.readouterr().out


def test_benchmark_returns_failure_when_latency_target_is_missed(tmp_path):
    image_directory = make_images(tmp_path)

    result = benchmark_main(
        [
            "--data-root",
            str(image_directory),
            "--image-count",
            "2",
            "--warmup-passes",
            "0",
            "--repeats",
            "1",
            "--max-ms",
            "0.000001",
        ]
    )

    assert result == 1
