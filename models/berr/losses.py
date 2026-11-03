"""Losses for end-to-end BERR training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.common.roi_utils import crop_and_resize


def charbonnier(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    error = torch.sqrt((pred - target).square() + 1e-6)
    if weight is None:
        return error.mean()
    while weight.ndim < error.ndim:
        weight = weight.unsqueeze(1)
    weight = weight.expand_as(error)
    return (error * weight).sum() / weight.sum().clamp_min(1e-6)


def flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    dx = (flow[..., :, 1:] - flow[..., :, :-1]).abs().mean()
    dy = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
    return dx + dy


def differentiable_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    c1, c2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(pred, 11, 1, 5)
    mu_y = F.avg_pool2d(target, 11, 1, 5)
    var_x = F.avg_pool2d(pred.square(), 11, 1, 5) - mu_x.square()
    var_y = F.avg_pool2d(target.square(), 11, 1, 5) - mu_y.square()
    cov = F.avg_pool2d(pred * target, 11, 1, 5) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2).clamp_min(1e-6)
    )
    return score.mean()


def compute_losses(
    model,
    outputs: dict[str, torch.Tensor],
    batch: dict,
    weights: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["target"]
    bbox = batch["roi_bbox"]
    gt_roi = crop_and_resize(target, bbox, model.roi_size)
    anchor_mask = outputs["degradation"][:, 0]
    all_degraded = outputs["all_degraded_roi"].detach()

    roi = charbonnier(outputs["restored_roi"], gt_roi)
    full = charbonnier(outputs["final"], target)
    visible = charbonnier(outputs["visible_roi"], gt_roi, 1.0 - all_degraded)
    prior = charbonnier(outputs["prior_roi"], gt_roi, all_degraded)
    outside = charbonnier(outputs["final"], outputs["anchor_full"], 1.0 - outputs["alpha_full"].detach())
    smooth = flow_smoothness(outputs["residual_flows"][:, 1:])
    ssim_loss = 1.0 - differentiable_ssim(outputs["final"], target)

    mask_loss = outputs["final"].new_zeros(())
    alignment = outputs["final"].new_zeros(())
    if "burst_masks" in batch:
        aligned_masks = model.align_supervision_masks(batch["burst_masks"], outputs, bbox).detach()
        pred = outputs["degradation"].clamp(1e-5, 1.0 - 1e-5)
        mask_loss = F.binary_cross_entropy(pred, aligned_masks) + F.l1_loss(pred, aligned_masks)
        target_stack = gt_roi[:, None].expand_as(outputs["aligned_rois"])
        alignment = charbonnier(outputs["aligned_rois"], target_stack, 1.0 - aligned_masks)
    elif "glare_mask" in batch:
        gt_mask = crop_and_resize(batch["glare_mask"], bbox, model.roi_size)
        pred = anchor_mask.clamp(1e-5, 1.0 - 1e-5)
        mask_loss = F.binary_cross_entropy(pred, gt_mask) + F.l1_loss(pred, gt_mask)

    values = {
        "roi": roi,
        "full": full,
        "ssim": ssim_loss,
        "mask": mask_loss,
        "visible": visible,
        "prior": prior,
        "outside": outside,
        "alignment": alignment,
        "flow_smooth": smooth,
    }
    total = sum(float(weights.get(name, 0.0)) * value for name, value in values.items())
    stats = {name: float(value.detach()) for name, value in values.items()}
    stats["total"] = float(total.detach())
    return total, stats
