"""Geometry-based tensor warping and landmark similarity transforms."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def estimate_similarity_transform_batch(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    """Estimate batched 2x3 similarity transforms (src -> dst).

    Args:
        src: (..., K, 2) source landmarks (x, y).
        dst: (..., K, 2) destination landmarks (x, y).

    Returns:
        (..., 2, 3) affine matrices.
    """
    if src.shape != dst.shape or src.ndim < 2 or src.shape[-1] != 2:
        raise ValueError(
            f"Expected src/dst with same shape (..., K, 2), got {src.shape} and {dst.shape}"
        )

    k = src.shape[-2]
    sx = src[..., :, 0]
    sy = src[..., :, 1]
    dx = dst[..., :, 0]
    dy = dst[..., :, 1]

    ones = torch.ones_like(sx)
    zeros = torch.zeros_like(sx)

    rows_x = torch.stack([sx, -sy, ones, zeros], dim=-1)
    rows_y = torch.stack([sy, sx, zeros, ones], dim=-1)

    A = torch.empty(*src.shape[:-2], 2 * k, 4, dtype=src.dtype, device=src.device)
    A[..., 0::2, :] = rows_x
    A[..., 1::2, :] = rows_y

    B = torch.empty(*src.shape[:-2], 2 * k, dtype=src.dtype, device=src.device)
    B[..., 0::2] = dx
    B[..., 1::2] = dy

    result = torch.linalg.lstsq(A, B).solution
    a, b, tx, ty = result[..., 0], result[..., 1], result[..., 2], result[..., 3]

    M = torch.empty(*src.shape[:-2], 2, 3, dtype=src.dtype, device=src.device)
    M[..., 0, 0] = a
    M[..., 0, 1] = -b
    M[..., 0, 2] = tx
    M[..., 1, 0] = b
    M[..., 1, 1] = a
    M[..., 1, 2] = ty
    return M


def estimate_similarity_transform(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    """Estimate a 2x3 similarity transform (translation + rotation + uniform scale).

    Uses least-squares on corresponding landmark pairs.

    Args:
        src: (K, 2) source landmarks (x, y).
        dst: (K, 2) destination landmarks (x, y).

    Returns:
        (2, 3) affine matrix mapping src -> dst.
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"Expected src/dst shape (K, 2), got {src.shape} and {dst.shape}")
    return estimate_similarity_transform_batch(src.unsqueeze(0), dst.unsqueeze(0)).squeeze(0)


def warp_frame(
    frame: torch.Tensor,
    M: torch.Tensor,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Apply a 2x3 similarity transform to a (C, H, W) image tensor via grid_sample.

    Args:
        frame: (C, H, W) image tensor.
        M: (2, 3) affine matrix (forward: src -> dst coordinates).
        output_size: (H_out, W_out); defaults to input size.

    Returns:
        (C, H_out, W_out) warped image.
    """
    C, H, W = frame.shape
    if output_size is None:
        output_size = (H, W)

    # grid_sample needs the *inverse* mapping: for each output pixel, where in input?
    # M maps src -> dst, so M_inv maps dst -> src.
    # Extend to 3x3 for inversion.
    M_full = torch.eye(3, dtype=M.dtype, device=M.device)
    M_full[:2, :] = M
    M_inv_full = torch.inverse(M_full)
    M_inv = M_inv_full[:2, :]

    # Convert to normalized coordinates [-1, 1] for grid_sample.
    # pixel (px, py) -> norm (2*px/(W-1)-1, 2*py/(H-1)-1)
    # We construct theta for affine_grid.
    # affine_grid expects theta that maps from [-1,1] output to [-1,1] input.
    # norm_out = theta * [x_in_norm, y_in_norm, 1]
    # We need to compose: pixel_out -> pixel_in via M_inv, then pixel_in -> norm_in.

    H_out, W_out = output_size

    # Normalization matrices
    # N_out: norm_out -> pixel_out: px = (nx+1)/2*(W_out-1), py = (ny+1)/2*(H_out-1)
    # N_in_inv: pixel_in -> norm_in
    N_out = torch.tensor(
        [[(W_out - 1) / 2.0, 0, (W_out - 1) / 2.0],
         [0, (H_out - 1) / 2.0, (H_out - 1) / 2.0],
         [0, 0, 1]],
        dtype=M.dtype, device=M.device,
    )
    N_in_inv = torch.tensor(
        [[2.0 / max(W - 1, 1), 0, -1],
         [0, 2.0 / max(H - 1, 1), -1],
         [0, 0, 1]],
        dtype=M.dtype, device=M.device,
    )

    theta_full = N_in_inv @ M_inv_full @ N_out
    theta = theta_full[:2, :].unsqueeze(0)

    grid = F.affine_grid(theta, (1, C, H_out, W_out), align_corners=True)
    warped = F.grid_sample(
        frame.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return warped.squeeze(0)


def warp_frames(
    frames: torch.Tensor,
    M: torch.Tensor,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Apply batched 2x3 similarity transforms to (N, C, H, W) frames."""
    if frames.ndim != 4 or M.ndim != 3 or M.shape[1:] != (2, 3) or frames.shape[0] != M.shape[0]:
        raise ValueError(
            f"Expected frames (N,C,H,W) and M (N,2,3), got {frames.shape} and {M.shape}"
        )

    N, C, H, W = frames.shape
    if output_size is None:
        output_size = (H, W)
    H_out, W_out = output_size

    M_full = torch.eye(3, dtype=M.dtype, device=M.device).unsqueeze(0).repeat(N, 1, 1)
    M_full[:, :2, :] = M
    M_inv_full = torch.linalg.inv(M_full)

    N_out = torch.tensor(
        [
            [(W_out - 1) / 2.0, 0.0, (W_out - 1) / 2.0],
            [0.0, (H_out - 1) / 2.0, (H_out - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=M.dtype,
        device=M.device,
    ).unsqueeze(0)
    N_in_inv = torch.tensor(
        [
            [2.0 / max(W - 1, 1), 0.0, -1.0],
            [0.0, 2.0 / max(H - 1, 1), -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=M.dtype,
        device=M.device,
    ).unsqueeze(0)

    theta_full = N_in_inv @ M_inv_full @ N_out
    theta = theta_full[:, :2, :]
    grid = F.affine_grid(theta, (N, C, H_out, W_out), align_corners=True)
    return F.grid_sample(
        frames, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def align_burst_to_anchor(
    burst: torch.Tensor,
    landmarks_per_frame: torch.Tensor,
    anchor_idx: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Align all burst frames to the anchor using landmark-based similarity transforms.

    Args:
        burst: (T, C, H, W) burst frames.
        landmarks_per_frame: (T, K, 2) landmarks for each frame.
        anchor_idx: which frame is the anchor.

    Returns:
        aligned_burst: (T, C, H, W) with anchor unchanged and others warped.
        transforms: list of T (2,3) matrices (identity for anchor).
    """
    T, C, H, W = burst.shape
    anchor_lm = landmarks_per_frame[anchor_idx]
    aligned = torch.empty_like(burst)
    transforms: list[torch.Tensor] = []

    for t in range(T):
        if t == anchor_idx:
            aligned[t] = burst[t]
            transforms.append(torch.eye(2, 3, dtype=burst.dtype, device=burst.device))
        else:
            M = estimate_similarity_transform(landmarks_per_frame[t], anchor_lm)
            aligned[t] = warp_frame(burst[t], M, (H, W))
            transforms.append(M)

    return aligned, transforms


def align_burst_to_anchor_batch(
    burst: torch.Tensor,
    landmarks: torch.Tensor,
    anchor_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align burst frames to each sample's anchor in a vectorized way.

    Args:
        burst: (B, T, C, H, W)
        landmarks: (B, T, K, 2)
        anchor_indices: (B,)

    Returns:
        aligned_burst: (B, T, C, H, W)
        transforms: (B, T, 2, 3) src->anchor transforms.
    """
    if burst.ndim != 5 or landmarks.ndim != 4:
        raise ValueError(f"Expected burst (B,T,C,H,W), landmarks (B,T,K,2), got {burst.shape}, {landmarks.shape}")
    if landmarks.shape[:2] != burst.shape[:2]:
        raise ValueError("landmarks leading dims must match burst (B,T)")

    B, T, C, H, W = burst.shape
    sample_idx = torch.arange(B, device=burst.device)
    anchor_lm = landmarks[sample_idx, anchor_indices]  # (B, K, 2)
    dst = anchor_lm[:, None, :, :].expand(-1, T, -1, -1)

    transforms = estimate_similarity_transform_batch(landmarks, dst)
    eye = torch.eye(2, 3, dtype=burst.dtype, device=burst.device)
    transforms[sample_idx, anchor_indices] = eye

    warped = warp_frames(
        burst.reshape(B * T, C, H, W),
        transforms.reshape(B * T, 2, 3),
        output_size=(H, W),
    ).reshape(B, T, C, H, W)
    warped[sample_idx, anchor_indices] = burst[sample_idx, anchor_indices]
    return warped, transforms
