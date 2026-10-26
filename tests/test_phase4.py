from pathlib import Path

import pytest
import torch
from PIL import Image

from experiments.infer_phase4 import main as infer_main
from experiments.infer_phase4 import tiled_predict
from experiments.run_phase4 import main as train_main
from models.moe_topk_restorer import MoETopKRestorer, ParallelDilatedAffineHead


@pytest.fixture
def tiny_image_directory(tmp_path: Path) -> Path:
    image_dir = tmp_path / "DIV2K_train_HR"
    image_dir.mkdir()
    for index in range(2):
        image = Image.new("RGB", (32, 32), color=(80 + index, 120, 160))
        image.save(image_dir / f"{index + 1:04d}.png")
    return image_dir


def test_moe_topk_restorer_shapes_gradients_and_auxiliary_loss():
    model = MoETopKRestorer(
        d=16,
        m=32,
        active_experts=2,
        top_k=8,
        noise_std=0,
    )
    low_resolution = torch.rand(2, 3, 8, 8)
    clean = torch.rand(2, 3, 32, 32)

    prediction, auxiliary = model(
        low_resolution, output_size=(32, 32), return_aux=True
    )
    loss = (prediction - clean).abs().mean() + auxiliary
    loss.backward()

    assert prediction.shape == clean.shape
    assert torch.isfinite(auxiliary)
    assert model.query_projection.weight.grad is not None
    assert torch.count_nonzero(model.query_projection.weight.grad).item() > 0
    assert model.dictionary.dictionary.grad is not None
    assert torch.count_nonzero(model.dictionary.dictionary.grad).item() > 0
    assert model.affine_head.proj[-1].weight.grad is not None
    assert torch.count_nonzero(model.affine_head.proj[-1].weight.grad).item() > 0


def test_parallel_dilated_head_is_identity_initialized_and_differentiable():
    head = ParallelDilatedAffineHead(hidden_dim=16)
    tokens = torch.randn(2, 16, 16, requires_grad=True)

    gamma, beta = head(tokens, height=4, width=4)
    modulated = gamma * tokens + beta

    assert gamma.shape == beta.shape == tokens.shape
    assert torch.equal(gamma, torch.ones_like(gamma))
    assert torch.equal(beta, torch.zeros_like(beta))

    torch.nn.init.normal_(head.proj[-1].weight)
    gamma, beta = head(tokens, height=4, width=4)
    modulated = gamma * tokens + beta
    modulated.sum().backward()
    assert head.proj[-1].weight.grad is not None
    assert torch.count_nonzero(head.proj[-1].weight.grad).item() > 0
    assert all(branch.weight.grad is not None for branch in head.branches)
    assert tokens.grad is not None


def test_parallel_dilated_head_validates_token_grid_and_dimensions():
    with pytest.raises(ValueError, match="hidden_dim must be at least 2"):
        ParallelDilatedAffineHead(hidden_dim=1)
    head = ParallelDilatedAffineHead(hidden_dim=16)
    with pytest.raises(ValueError, match="token count 5 must equal height\\*width"):
        head(torch.randn(1, 5, 16), height=2, width=2)
    with pytest.raises(ValueError, match="expected tokens with shape"):
        head(torch.randn(1, 16, 8), height=4, width=4)


def test_tiled_inference_covers_image_and_preserves_shape():
    model = MoETopKRestorer(d=16, m=32, top_k=8, active_experts=2)
    image = torch.rand(3, 29, 37)

    prediction = tiled_predict(
        model, image, torch.device("cpu"), tile_size=16, overlap=4
    )

    assert prediction.shape == image.shape
    assert torch.isfinite(prediction).all()
    assert torch.all((prediction >= 0) & (prediction <= 1))


def test_phase4_training_and_checkpoint_inference_smoke(
    tiny_image_directory, tmp_path
):
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "inference"

    assert train_main(
        [
            "--data-root",
            str(tiny_image_directory),
            "--degradation",
            "noise",
            "--image-count",
            "2",
            "--epochs",
            "2",
            "--patch-size",
            "16",
            "--dim",
            "16",
            "--dictionary-size",
            "32",
            "--top-k",
            "8",
            "--output-dir",
            str(checkpoint_dir),
            "--min-psnr",
            "0",
            "--device",
            "cpu",
        ]
    ) == 0
    assert (checkpoint_dir / "moe_topk_restorer.pt").is_file()
    assert (checkpoint_dir / "metrics.json").is_file()

    assert infer_main(
        [
            "--data-root",
            str(tiny_image_directory),
            "--checkpoint",
            str(checkpoint_dir / "moe_topk_restorer.pt"),
            "--image-count",
            "2",
            "--tile-size",
            "16",
            "--overlap",
            "4",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ]
    ) == 0
    output_images = sorted(output_dir.glob("*_restored.png"))
    assert len(output_images) == 2
    with Image.open(output_images[0]) as image:
        assert image.size == (32, 32)
    assert (output_dir / "inference_metrics.json").is_file()


def test_model_rejects_non_rgb_images():
    model = MoETopKRestorer(d=16, m=32, top_k=8)
    with pytest.raises(ValueError, match="expected RGB image"):
        model(torch.rand(1, 1, 16, 16))
