"""Four-frame RGB, landmark, and eye-ROI preprocessing for BERR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:  # Pillow remains a supported fallback.
    cv2 = None

from datasets.face_geometry import (
    compute_roi_bbox_from_landmarks,
    extract_landmarks,
    render_geometry_heatmap_roi,
)


class VisionInputError(ValueError):
    """Base class for cleanly reportable vision input errors."""


class InsufficientFramesError(VisionInputError):
    """Raised unless exactly four frames are supplied."""


class NoFaceError(VisionInputError):
    """Raised when landmarks cannot be detected in one of the face frames."""


class PreprocessingError(VisionInputError):
    """Raised for malformed images, landmarks, or ROI data."""


@dataclass
class PreparedBurst:
    burst: torch.Tensor
    landmarks: torch.Tensor
    roi_bbox: torch.Tensor
    geometry_heatmap_roi: torch.Tensor
    rgb_frames: list[np.ndarray]
    original_shapes: list[tuple[int, int]]


def _as_rgb_array(frame) -> np.ndarray:
    if isinstance(frame, (str, Path)):
        with Image.open(frame) as image:
            return np.asarray(image.convert("RGB"))
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"))
    if torch.is_tensor(frame):
        value = frame.detach().cpu()
        if value.ndim != 3:
            raise PreprocessingError(f"Each tensor frame must be 3D, got {tuple(value.shape)}")
        if value.shape[0] in (1, 3, 4):
            value = value.permute(1, 2, 0)
        frame = value.numpy()
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[2] == 3 and array.dtype == np.uint8:
        return array if array.flags.c_contiguous else np.ascontiguousarray(array)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise PreprocessingError(f"Expected HxWx3 RGB image, got shape {array.shape}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise PreprocessingError("Frame contains NaN or infinite values")
        if float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _frame_list(frames, expected_frames: int) -> list[np.ndarray]:
    if torch.is_tensor(frames):
        if frames.ndim != 4:
            raise PreprocessingError(
                f"Burst tensor must have shape [T,3,H,W] or [T,H,W,3], got {tuple(frames.shape)}"
            )
        items = [frames[index] for index in range(frames.shape[0])]
    elif isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
        items = list(frames)
    else:
        raise PreprocessingError("frames must be a four-item sequence or a 4D tensor")
    if len(items) != expected_frames:
        raise InsufficientFramesError(
            f"Expected exactly {expected_frames} frames, received {len(items)}"
        )
    return [_as_rgb_array(item) for item in items]


def _resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    if image.shape[:2] == (image_size, image_size):
        return image if image.flags.c_contiguous else np.ascontiguousarray(image)
    if cv2 is not None:
        return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return np.asarray(
        Image.fromarray(image, mode="RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
    )


def _prepare_landmarks(
    resized: list[np.ndarray],
    original_shapes: list[tuple[int, int]],
    landmarks,
    image_size: int,
    face_landmarker_path: str | Path | None,
    anchor_landmarks=None,
    reuse_anchor_landmarks: bool = True,
) -> np.ndarray:
    if landmarks is None:
        if anchor_landmarks is not None:
            anchor = np.asarray(anchor_landmarks, dtype=np.float32).copy()
            height, width = original_shapes[0]
            anchor[:, 0] *= image_size / float(width)
            anchor[:, 1] *= image_size / float(height)
        else:
            anchor = extract_landmarks(resized[0], model_path=face_landmarker_path)
        if anchor is None:
            raise NoFaceError("No face detected in frame 0")
        if reuse_anchor_landmarks:
            return np.repeat(np.asarray(anchor, dtype=np.float32)[None], len(resized), axis=0)

        detected = [anchor]
        for index, image in enumerate(resized[1:], start=1):
            points = extract_landmarks(image, model_path=face_landmarker_path)
            if points is None:
                raise NoFaceError(f"No face detected in frame {index}")
            detected.append(points)
        return np.stack(detected).astype(np.float32)

    points = np.asarray(
        landmarks.detach().cpu().numpy() if torch.is_tensor(landmarks) else landmarks,
        dtype=np.float32,
    )
    if points.ndim == 2 and points.shape[-1] == 2:
        points = np.repeat(points[None], len(resized), axis=0)
    if points.ndim != 3 or points.shape[0] != len(resized) or points.shape[-1] != 2:
        raise PreprocessingError(
            f"landmarks must have shape [4,K,2] or [K,2], got {points.shape}"
        )
    points = points.copy()
    for index, (height, width) in enumerate(original_shapes):
        points[index, :, 0] *= image_size / float(width)
        points[index, :, 1] *= image_size / float(height)
    return points


def prepare_burst(
    frames,
    *,
    landmarks=None,
    roi_bbox=None,
    expected_frames: int = 4,
    image_size: int = 256,
    roi_size: int = 128,
    face_landmarker_path: str | Path | None = None,
    anchor_landmarks=None,
    reuse_anchor_landmarks: bool = True,
) -> PreparedBurst:
    """Validate and convert four RGB face frames into one BERR input batch."""
    originals = _frame_list(frames, expected_frames)
    shapes = [image.shape[:2] for image in originals]
    resized = [_resize_rgb(image, image_size) for image in originals]
    points = _prepare_landmarks(
        resized,
        shapes,
        landmarks,
        image_size,
        face_landmarker_path,
        anchor_landmarks,
        reuse_anchor_landmarks,
    )

    if roi_bbox is None:
        bbox = compute_roi_bbox_from_landmarks(points[0], image_size, image_size)
    else:
        raw_bbox = np.asarray(
            roi_bbox.detach().cpu().numpy() if torch.is_tensor(roi_bbox) else roi_bbox,
            dtype=np.float32,
        ).reshape(-1)
        if raw_bbox.size != 4 or not np.isfinite(raw_bbox).all():
            raise PreprocessingError("roi_bbox must contain four finite xyxy values")
        anchor_h, anchor_w = shapes[0]
        raw_bbox[[0, 2]] *= image_size / float(anchor_w)
        raw_bbox[[1, 3]] *= image_size / float(anchor_h)
        x0, y0, x1, y1 = raw_bbox.tolist()
        bbox = (
            int(np.clip(x0, 0, image_size - 1)),
            int(np.clip(y0, 0, image_size - 1)),
            int(np.clip(x1, 1, image_size)),
            int(np.clip(y1, 1, image_size)),
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise PreprocessingError(f"Invalid roi_bbox after resizing: {bbox}")

    geometry = render_geometry_heatmap_roi(points[0], bbox, roi_size)
    # One contiguous batch conversion avoids four image copies plus torch.stack.
    batch = np.stack(resized, axis=0)
    burst = torch.from_numpy(batch).permute(0, 3, 1, 2).to(torch.float32).mul_(1.0 / 255.0)
    return PreparedBurst(
        burst=burst.unsqueeze(0),
        landmarks=torch.from_numpy(points).unsqueeze(0),
        roi_bbox=torch.tensor(bbox, dtype=torch.long).unsqueeze(0),
        geometry_heatmap_roi=geometry.unsqueeze(0),
        rgb_frames=resized,
        original_shapes=shapes,
    )
