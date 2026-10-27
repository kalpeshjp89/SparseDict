"""Shared metrics and tiled prediction helpers for Phase 4."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.moe_topk_restorer import MoETopKRestorer


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def gradient_difference_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Compare horizontal and vertical image gradients to encourage sharp detail."""
    losses = []
    if prediction.size(-1) > 1:
        losses.append(
            F.l1_loss(
                prediction[..., 1:] - prediction[..., :-1],
                target[..., 1:] - target[..., :-1],
            )
        )
    if prediction.size(-2) > 1:
        losses.append(
            F.l1_loss(
                prediction[..., 1:, :] - prediction[..., :-1, :],
                target[..., 1:, :] - target[..., :-1, :],
            )
        )
    if not losses:
        return prediction.new_zeros(())
    return torch.stack(losses).mean()


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.inference_mode()
def tiled_predict(
    model: MoETopKRestorer,
    image: torch.Tensor,
    device: torch.device,
    tile_size: int = 128,
    overlap: int = 16,
) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"expected image tensor (C,H,W), got {tuple(image.shape)}")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError(
            f"require tile_size>0 and 0<=overlap<tile_size; "
            f"got tile_size={tile_size}, overlap={overlap}"
        )
    _, height, width = image.shape
    output = torch.zeros_like(image)
    weights = image.new_zeros((1, height, width))
    row_starts = _tile_starts(height, tile_size, overlap)
    column_starts = _tile_starts(width, tile_size, overlap)
    model.eval()
    for top in row_starts:
        bottom = min(top + tile_size, height)
        for left in column_starts:
            right = min(left + tile_size, width)
            tile = image[:, top:bottom, left:right].unsqueeze(0).to(device)
            prediction = model(tile)[0].to(image.device)
            output[:, top:bottom, left:right] += prediction
            weights[:, top:bottom, left:right] += 1
    return (output / weights.clamp_min(1)).clamp(0, 1)
