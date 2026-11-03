"""Deterministic, lens-confined glare stress test for labelled FER images."""

from __future__ import annotations

import hashlib
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from datasets.face_geometry import compute_roi_bbox_from_landmarks, extract_landmarks

SEVERITY_ALPHA = {"mild": (0.24, 0.34), "medium": (0.42, 0.56), "strong": (0.60, 0.74)}
LEFT_EYE = [33, 133, 160, 158, 153, 144]
RIGHT_EYE = [362, 263, 387, 385, 380, 373]
TRANSFORMS = (
    (0.0, 0.0, 0.0, 1.000),
    (1.0, -0.7, 0.35, 1.003),
    (-0.8, 0.9, -0.30, 0.997),
    (0.6, 0.4, 0.20, 1.002),
)


def _seed(sample_id: str, severity: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{sample_id}:{severity}".encode()).digest()[:4], "little")


def _affine(width: int, height: int, dx: float, dy: float, angle: float, scale: float):
    """Return forward 3x3 and PIL inverse affine coefficients."""
    radians = math.radians(angle)
    matrix = scale * np.array(
        [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]],
        dtype=np.float64,
    )
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    offset = center + np.array([dx, dy]) - matrix @ center
    forward = np.eye(3, dtype=np.float64)
    forward[:2, :2] = matrix
    forward[:2, 2] = offset
    inverse = np.linalg.inv(forward)
    return forward, tuple(inverse[:2].reshape(-1))


def _transform_points(points: np.ndarray, forward: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points.astype(np.float64), np.ones((len(points), 1))], axis=1)
    return (homogeneous @ forward.T)[:, :2].astype(np.float32)


def _lens_ellipse(points: np.ndarray, indices: list[int], width: int, height: int):
    eye = points[indices]
    x_min, y_min = eye.min(axis=0)
    x_max, y_max = eye.max(axis=0)
    eye_width = max(float(x_max - x_min), width * 0.035)
    eye_height = max(float(y_max - y_min), height * 0.012)
    cx = float((x_min + x_max) / 2)
    cy = float((y_min + y_max) / 2 - 0.10 * eye_height)
    radius_x = 0.72 * eye_width
    radius_y = max(1.9 * eye_height, 0.30 * eye_width)
    return (cx - radius_x, cy - radius_y, cx + radius_x, cy + radius_y)


def _lens_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
    hard = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(hard)
    for indices in (LEFT_EYE, RIGHT_EYE):
        draw.ellipse(_lens_ellipse(points, indices, width, height), fill=255)
    feathered = hard.filter(ImageFilter.GaussianBlur(max(0.8, min(width, height) * 0.004)))
    mask = np.asarray(feathered, dtype=np.float32) / 255.0
    mask[mask < 1.0 / 255.0] = 0.0
    return mask


def _glare_pattern(
    points: np.ndarray,
    width: int,
    height: int,
    frame_index: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layer)
    phase = frame_index - 1.5
    for eye_index, indices in enumerate((LEFT_EYE, RIGHT_EYE)):
        box = _lens_ellipse(points, indices, width, height)
        x0, y0, x1, y1 = box
        lens_w, lens_h = x1 - x0, y1 - y0
        direction = 1 if eye_index == 0 else -1
        drift_x = phase * 0.045 * lens_w * direction
        drift_y = phase * 0.025 * lens_h
        base_y = y0 + lens_h * (0.35 + rng.uniform(-0.04, 0.04)) + drift_y
        line_width = max(3, int(lens_h * rng.uniform(0.28, 0.38)))
        draw.line(
            (x0 + 0.10 * lens_w + drift_x, base_y, x1 - 0.08 * lens_w + drift_x, base_y + 0.28 * lens_h),
            fill=245,
            width=line_width,
        )
        blob_cx = x0 + lens_w * (0.66 + 0.04 * phase * direction)
        blob_cy = y0 + lens_h * (0.48 + 0.025 * phase)
        blob_rx, blob_ry = 0.24 * lens_w, 0.18 * lens_h
        draw.ellipse((blob_cx - blob_rx, blob_cy - blob_ry, blob_cx + blob_rx, blob_cy + blob_ry), fill=185)
    pattern = layer.filter(ImageFilter.GaussianBlur(max(1.8, min(width, height) * 0.014)))
    return np.asarray(pattern, dtype=np.float32) / 255.0


def synthesize_glare_burst(image, sample_id: str, severity: str, *, landmarks=None, model_path=None):
    """Create four transformed clean/glare frames with glare confined to virtual lenses."""
    if severity not in SEVERITY_ALPHA:
        raise ValueError(f"Unknown severity: {severity}")
    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.uint8)
    h, w = rgb.shape[:2]
    points = np.asarray(landmarks if landmarks is not None else extract_landmarks(rgb, model_path), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] <= max(RIGHT_EYE):
        raise ValueError("No usable face landmarks")
    rng = np.random.RandomState(_seed(sample_id, severity))
    lo, hi = SEVERITY_ALPHA[severity]
    base_alpha = rng.uniform(lo, hi)
    frames, clean_frames, all_points, masks, lens_masks = [], [], [], [], []
    for frame_index, (dx, dy, angle, scale) in enumerate(TRANSFORMS):
        forward, inverse_coefficients = _affine(w, h, dx, dy, angle, scale)
        clean = np.asarray(
            Image.fromarray(rgb).transform(
                (w, h), Image.Transform.AFFINE, inverse_coefficients,
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
        transformed_points = _transform_points(points, forward)
        lens = _lens_mask(transformed_points, w, h)
        pattern = _glare_pattern(transformed_points, w, h, frame_index, rng)
        glare_mask = lens * pattern
        glare_mask[glare_mask < 1.0 / 255.0] = 0.0
        temporal_scale = (0.92, 1.00, 1.07, 0.97)[frame_index]
        alpha = np.clip(base_alpha * temporal_scale * glare_mask, 0.0, 0.75)[..., None]
        tint = np.array([244, 250, 255], dtype=np.float32)
        output = np.clip(clean.astype(np.float32) * (1.0 - alpha) + tint * alpha, 0, 255).round().astype(np.uint8)
        frames.append(output)
        clean_frames.append(clean)
        all_points.append(transformed_points)
        masks.append(glare_mask)
        lens_masks.append(lens)
    anchor_bbox = compute_roi_bbox_from_landmarks(all_points[0], h, w, padding_ratio=0.12)
    return {
        "frames": frames,
        "clean_frames": clean_frames,
        "landmarks": np.stack(all_points),
        "roi_bbox": anchor_bbox,
        "masks": np.stack(masks),
        "lens_masks": np.stack(lens_masks),
        "severity": severity,
        "seed": _seed(sample_id, severity),
    }
