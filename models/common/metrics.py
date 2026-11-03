"""Basic metrics used by smoke tests and early baselines."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .mask_utils import threshold_mask
from .roi_utils import crop_and_resize


def psnr(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((prediction - target) ** 2).item()
    if mse <= eps:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _ssim_unary_map(x: torch.Tensor, y: torch.Tensor, window_size: int, data_range: float) -> torch.Tensor:
    """SSIM map for single-channel tensors (N,1,H,W). Returns map (N,1,H',W')."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    pad = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x.clamp_min(0.0) + sigma_y.clamp_min(0.0) + c2)
    return num / den.clamp_min(1e-12)


def ssim(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11, data_range: float = 1.0) -> float:
    """Mean SSIM over batch and RGB channels. prediction/target: N x 3 x H x W in [0, 1]."""
    if prediction.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    n, c, _, _ = prediction.shape
    if n == 0:
        return 0.0
    x = prediction.clamp(0.0, 1.0)
    y = target.clamp(0.0, 1.0)
    total = 0.0
    count = 0
    ws = min(window_size, x.size(-1), x.size(-2))
    if ws % 2 == 0:
        ws -= 1
    if ws < 3:
        ws = 3
    for i in range(n):
        for ch in range(c):
            xm = x[i : i + 1, ch : ch + 1]
            ym = y[i : i + 1, ch : ch + 1]
            smap = _ssim_unary_map(xm, ym, ws, data_range)
            total += float(smap.mean().item())
            count += 1
    return total / max(count, 1)


def mask_iou(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = threshold_mask(prediction, threshold)
    tgt = threshold_mask(target, threshold)
    intersection = torch.sum(pred * tgt).item()
    union = torch.sum((pred + tgt) > 0).item()
    return float(intersection / union) if union > 0 else 1.0


def mask_f1(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = threshold_mask(prediction, threshold)
    tgt = threshold_mask(target, threshold)
    tp = torch.sum(pred * tgt).item()
    precision_den = torch.sum(pred).item()
    recall_den = torch.sum(tgt).item()
    precision = tp / precision_den if precision_den > 0 else 1.0
    recall = tp / recall_den if recall_den > 0 else 1.0
    denom = precision + recall
    return float((2.0 * precision * recall / denom) if denom > 0 else 0.0)


def roi_l1(prediction: torch.Tensor, target: torch.Tensor, bboxes: torch.Tensor, roi_size: int = 128) -> float:
    pred_roi = crop_and_resize(prediction, bboxes, roi_size)
    target_roi = crop_and_resize(target, bboxes, roi_size)
    return torch.mean(torch.abs(pred_roi - target_roi)).item()
