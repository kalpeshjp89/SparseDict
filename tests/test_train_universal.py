import torch

from experiments.train_universal import _metrics, _ssim


def test_ssim_is_finite_and_one_for_identical_images():
    image = torch.rand(1, 3, 8, 10)

    score = _ssim(image, image)

    assert torch.isfinite(score)
    assert torch.allclose(score, torch.tensor(1.0), atol=1e-6)


def test_validation_metrics_report_psnr_ssim_and_l1():
    target = torch.rand(1, 3, 16, 16)
    metrics = _metrics(target, target)

    assert set(metrics) == {"psnr", "ssim", "l1"}
    assert metrics["psnr"] == 120.0
    assert metrics["ssim"] == 1.0
    assert metrics["l1"] == 0.0
