"""Face landmarks, eye ROI boxes, and geometry heatmap rendering.

Uses MediaPipe FaceLandmarker when available; otherwise a CelebA-aligned heuristic
with optional OpenCV face validation (for environments where MediaPipe is missing).
"""

from __future__ import annotations

import logging
import atexit
import threading
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

_log = logging.getLogger(__name__)

EYE_AND_BROW_INDICES = [
    33, 133, 160, 158, 153, 144,
    362, 263, 387, 385, 380, 373,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
]

KEY_LANDMARK_INDICES = [33, 133, 362, 263, 70, 300, 168, 6]

_MEDIAPIPE_AVAILABLE: bool | None = None
_MEDIAPIPE_LANDMARKERS: dict[str, tuple[object, threading.Lock]] = {}
_OPENCV_FACE_CASCADES: dict[str, object] = {}


def _check_mediapipe() -> bool:
    global _MEDIAPIPE_AVAILABLE
    if _MEDIAPIPE_AVAILABLE is None:
        try:
            import mediapipe  # noqa: F401
            _MEDIAPIPE_AVAILABLE = True
        except ImportError:
            _MEDIAPIPE_AVAILABLE = False
            _log.info("mediapipe not available; using heuristic landmark fallback.")
    return _MEDIAPIPE_AVAILABLE


def _celeba_aligned_landmarks(h: int, w: int) -> np.ndarray:
    """Generate 478 synthetic landmarks for a CelebA-aligned crop."""
    lm = np.zeros((478, 2), dtype=np.float32)

    cx, cy_eyes = w * 0.5, h * 0.38
    eye_half_w = w * 0.14

    le_cx, le_cy = cx - eye_half_w, cy_eyes
    lm[33] = [le_cx - w * 0.05, le_cy]
    lm[133] = [le_cx + w * 0.05, le_cy]
    lm[160] = [le_cx - w * 0.03, le_cy - h * 0.015]
    lm[158] = [le_cx + w * 0.02, le_cy - h * 0.015]
    lm[153] = [le_cx + w * 0.03, le_cy + h * 0.012]
    lm[144] = [le_cx - w * 0.02, le_cy + h * 0.012]

    re_cx, re_cy = cx + eye_half_w, cy_eyes
    lm[362] = [re_cx - w * 0.05, re_cy]
    lm[263] = [re_cx + w * 0.05, re_cy]
    lm[387] = [re_cx - w * 0.03, re_cy - h * 0.015]
    lm[385] = [re_cx + w * 0.02, re_cy - h * 0.015]
    lm[380] = [re_cx + w * 0.03, re_cy + h * 0.012]
    lm[373] = [re_cx - w * 0.02, re_cy + h * 0.012]

    lb_cy = cy_eyes - h * 0.06
    lm[70] = [le_cx - w * 0.06, lb_cy]
    lm[63] = [le_cx - w * 0.03, lb_cy - h * 0.01]
    lm[105] = [le_cx, lb_cy - h * 0.012]
    lm[66] = [le_cx + w * 0.03, lb_cy - h * 0.008]
    lm[107] = [le_cx + w * 0.06, lb_cy]

    rb_cy = cy_eyes - h * 0.06
    lm[336] = [re_cx - w * 0.06, rb_cy]
    lm[296] = [re_cx - w * 0.03, rb_cy - h * 0.01]
    lm[334] = [re_cx, rb_cy - h * 0.012]
    lm[293] = [re_cx + w * 0.03, rb_cy - h * 0.008]
    lm[300] = [re_cx + w * 0.06, rb_cy]

    lm[168] = [cx, cy_eyes + h * 0.04]
    lm[6] = [cx, cy_eyes + h * 0.12]

    used = set(EYE_AND_BROW_INDICES + KEY_LANDMARK_INDICES)
    rng = np.random.RandomState(0)
    for i in range(478):
        if i not in used:
            lm[i] = [
                cx + rng.uniform(-w * 0.3, w * 0.3),
                h * 0.5 + rng.uniform(-h * 0.35, h * 0.35),
            ]

    return lm


def _extract_landmarks_mediapipe(
    image_rgb: np.ndarray,
    model_path: str | Path | None = None,
) -> np.ndarray | None:
    """MediaPipe FaceLandmarker -> (478, 2) pixel coords or None."""
    import mediapipe as mp

    if model_path is None:
        model_path = str(
            Path(__file__).resolve().parents[1] / "assets" / "face_landmarker.task"
        )

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    cache_key = str(Path(model_path).resolve())
    cached = _MEDIAPIPE_LANDMARKERS.get(cache_key)
    if cached is None:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
        )
        cached = (FaceLandmarker.create_from_options(options), threading.Lock())
        _MEDIAPIPE_LANDMARKERS[cache_key] = cached
    landmarker, lock = cached
    with lock:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    h, w = image_rgb.shape[:2]
    lms = result.face_landmarks[0]
    return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)


def _close_mediapipe_landmarkers() -> None:
    for landmarker, _ in _MEDIAPIPE_LANDMARKERS.values():
        try:
            landmarker.close()
        except Exception:
            pass
    _MEDIAPIPE_LANDMARKERS.clear()


atexit.register(_close_mediapipe_landmarkers)


def _extract_landmarks_heuristic(
    image_rgb: np.ndarray,
    jitter_seed: int | None = None,
) -> np.ndarray | None:
    """Detector-validated canonical landmarks for low-resolution aligned faces.

    This fallback never emits landmarks unless an OpenCV face cascade first
    returns a plausible, centrally located face box.  The canonical geometry is
    then fitted inside that detected box; it is not a no-face default.
    """
    detected = _detect_face_bbox_opencv(image_rgb)
    if detected is None:
        return None
    h, w = image_rgb.shape[:2]
    lm = _celeba_aligned_landmarks(h, w)

    if jitter_seed is not None:
        rng = np.random.RandomState(jitter_seed)
        jitter = rng.randn(478, 2).astype(np.float32) * 1.5
        used = set(EYE_AND_BROW_INDICES + KEY_LANDMARK_INDICES)
        for i in used:
            lm[i] += jitter[i]
        lm[:, 0] = np.clip(lm[:, 0], 0, w - 1)
        lm[:, 1] = np.clip(lm[:, 1], 0, h - 1)

    return lm


def _opencv_cascade(name: str):
    import cv2

    detector = _OPENCV_FACE_CASCADES.get(name)
    if detector is None:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if detector.empty():
            raise RuntimeError(f"Unable to load OpenCV face cascade: {name}")
        _OPENCV_FACE_CASCADES[name] = detector
    return detector


def _landmarks_from_face_bbox(bbox: Sequence[int]) -> np.ndarray:
    x, y, width, height = [int(value) for value in bbox]
    landmarks = _celeba_aligned_landmarks(max(height, 1), max(width, 1))
    landmarks[:, 0] += x
    landmarks[:, 1] += y
    return landmarks


def _detect_face_bbox_opencv(
    image_rgb: np.ndarray,
    *,
    detection_size: int = 384,
) -> tuple[tuple[int, int, int, int], str] | None:
    """Return a validated face box and detector method, or ``None``.

    Low-resolution inputs are enlarged for detection only.  The ordered cascade
    is deliberately conservative first (frontal, three neighbours), then adds
    profile and contrast-enhanced fallbacks.  Candidate boxes must satisfy area,
    centrality, and aspect-ratio checks before they are accepted.
    """
    try:
        import cv2
    except ImportError:
        return None

    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        return None
    native_h, native_w = image.shape[:2]
    scale = max(1.0, detection_size / float(max(native_h, native_w)))
    detect_w = max(1, int(round(native_w * scale)))
    detect_h = max(1, int(round(native_h * scale)))
    if scale > 1.0:
        image = cv2.resize(image, (detect_w, detect_h), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    variants = {
        "raw": gray,
        "equalized": cv2.equalizeHist(gray),
        "clahe": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray),
    }
    cascades = {
        "haar_default": "haarcascade_frontalface_default.xml",
        "haar_alt2": "haarcascade_frontalface_alt2.xml",
        "haar_profile": "haarcascade_profileface.xml",
    }
    attempts = (
        ("haar_default", "raw", 3, False),
        ("haar_default", "equalized", 3, False),
        ("haar_alt2", "raw", 3, False),
        ("haar_alt2", "equalized", 3, False),
        ("haar_profile", "raw", 3, False),
        ("haar_profile", "raw", 3, True),
        ("haar_default", "clahe", 2, False),
        ("haar_alt2", "clahe", 2, False),
        ("haar_profile", "clahe", 2, False),
        ("haar_profile", "clahe", 2, True),
    )
    min_side = max(24, int(round(min(detect_h, detect_w) * 0.15625)))
    for cascade_key, variant_key, neighbours, flipped in attempts:
        source = cv2.flip(variants[variant_key], 1) if flipped else variants[variant_key]
        faces = _opencv_cascade(cascades[cascade_key]).detectMultiScale(
            source,
            scaleFactor=1.08,
            minNeighbors=neighbours,
            minSize=(min_side, min_side),
        )
        valid = []
        for raw_x, raw_y, raw_w, raw_h in faces:
            x = detect_w - (raw_x + raw_w) if flipped else raw_x
            area = (raw_w * raw_h) / float(detect_w * detect_h)
            center_x = (x + raw_w / 2.0) / detect_w
            center_y = (raw_y + raw_h / 2.0) / detect_h
            aspect = raw_w / max(raw_h, 1)
            if area >= 0.08 and 0.18 <= center_x <= 0.82 and 0.16 <= center_y <= 0.82 and 0.55 <= aspect <= 1.55:
                centrality = abs(center_x - 0.5) + abs(center_y - 0.48)
                valid.append((area - 0.1 * centrality, x, raw_y, raw_w, raw_h))
        if valid:
            _, x, y, width, height = max(valid)
            native_bbox = (
                int(round(x / scale)),
                int(round(y / scale)),
                max(1, int(round(width / scale))),
                max(1, int(round(height / scale))),
            )
            suffix = "_flipped" if flipped else ""
            method = f"{cascade_key}_{variant_key}_n{neighbours}{suffix}"
            return native_bbox, method
    return None


def extract_landmarks_with_fallback(
    image_rgb: np.ndarray,
    model_path: str | Path | None = None,
    *,
    upsample_sizes: Sequence[int] = (256, 384, 512),
    aligned_face_crop: bool = True,
) -> tuple[np.ndarray | None, str]:
    """Detect landmarks with ordered native, upsampled, then local fallbacks.

    When a local cascade validates a face in an already aligned face crop, the
    full-crop template preserves the dataset's eye geometry.  With
    ``aligned_face_crop=False``, the template is fitted inside the detected box.
    No template is returned when all face detectors fail.
    """
    image = np.asarray(image_rgb, dtype=np.uint8)
    native_h, native_w = image.shape[:2]
    if _check_mediapipe():
        try:
            points = _extract_landmarks_mediapipe(image, model_path)
        except Exception:
            points = None
        if points is not None:
            return points, "native_mediapipe"
        from PIL import Image

        pil_image = Image.fromarray(image, mode="RGB")
        for size in upsample_sizes:
            if size <= max(native_h, native_w):
                continue
            detected = np.asarray(
                pil_image.resize((size, size), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )
            try:
                points = _extract_landmarks_mediapipe(detected, model_path)
            except Exception:
                points = None
            if points is not None:
                points = np.asarray(points, dtype=np.float32).copy()
                points[:, 0] *= native_w / float(size)
                points[:, 1] *= native_h / float(size)
                return points, f"upsampled_mediapipe_{size}"

    detected_face = _detect_face_bbox_opencv(image)
    if detected_face is not None:
        bbox, detector_method = detected_face
        if aligned_face_crop:
            points = _celeba_aligned_landmarks(native_h, native_w)
            method = f"opencv_validated_aligned_{detector_method}"
        else:
            points = _landmarks_from_face_bbox(bbox)
            method = f"opencv_bbox_{detector_method}"
        return points, method
    return None, "no_face"


def extract_landmarks(
    image_rgb: np.ndarray,
    model_path: str | Path | None = None,
    jitter_seed: int | None = None,
) -> np.ndarray | None:
    """478 landmarks using detector-validated MediaPipe/OpenCV fallbacks."""
    landmarks, _ = extract_landmarks_with_fallback(image_rgb, model_path)
    if landmarks is not None and jitter_seed is not None:
        rng = np.random.RandomState(jitter_seed)
        jitter = rng.randn(478, 2).astype(np.float32) * 1.5
        for index in set(EYE_AND_BROW_INDICES + KEY_LANDMARK_INDICES):
            landmarks[index] += jitter[index]
    return landmarks


# prepare script imports this name
extract_landmarks_mediapipe = extract_landmarks


def compute_roi_bbox_from_landmarks(
    landmarks: np.ndarray | torch.Tensor,
    image_h: int,
    image_w: int,
    padding_ratio: float = 0.25,
    indices: Sequence[int] | None = None,
) -> tuple[int, int, int, int]:
    if indices is None:
        indices = EYE_AND_BROW_INDICES

    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.detach().cpu().numpy()

    pts = landmarks[indices]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    bw = x_max - x_min
    bh = y_max - y_min
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio

    x0 = int(max(0, x_min - pad_x))
    y0 = int(max(0, y_min - pad_y))
    x1 = int(min(image_w, x_max + pad_x))
    y1 = int(min(image_h, y_max + pad_y))

    if x1 <= x0:
        x1 = min(x0 + 1, image_w)
    if y1 <= y0:
        y1 = min(y0 + 1, image_h)

    return x0, y0, x1, y1


def render_geometry_heatmap(
    landmarks: np.ndarray | torch.Tensor,
    height: int,
    width: int,
    sigma: float = 3.0,
    indices: Sequence[int] | None = None,
) -> torch.Tensor:
    if indices is None:
        indices = KEY_LANDMARK_INDICES

    if isinstance(landmarks, torch.Tensor):
        pts = landmarks.detach().float()
    else:
        pts = torch.from_numpy(landmarks).float()

    pts = pts[indices]
    k = pts.shape[0]

    yy = torch.arange(height, dtype=torch.float32).unsqueeze(1)
    xx = torch.arange(width, dtype=torch.float32).unsqueeze(0)

    two_sig2 = max(2.0 * sigma * sigma, 1e-6)
    acc = torch.zeros(height, width, dtype=torch.float32)
    for i in range(k):
        px, py = pts[i, 0].item(), pts[i, 1].item()
        acc += torch.exp(-((xx - px) ** 2 + (yy - py) ** 2) / two_sig2)

    vmax = acc.max()
    if vmax > 1e-6:
        acc = acc / vmax
    return acc.unsqueeze(0).clamp(0.0, 1.0)


def render_geometry_heatmap_roi(
    landmarks: np.ndarray | torch.Tensor,
    roi_bbox: tuple[int, int, int, int] | Sequence[int],
    roi_size: int,
    sigma: float = 3.0,
    indices: Sequence[int] | None = None,
) -> torch.Tensor:
    if isinstance(landmarks, torch.Tensor):
        lm = landmarks.detach().cpu().numpy().copy()
    else:
        lm = np.array(landmarks, dtype=np.float32).copy()

    x0, y0, x1, y1 = [int(v) for v in roi_bbox]
    roi_w = max(x1 - x0, 1)
    roi_h = max(y1 - y0, 1)

    lm[:, 0] = (lm[:, 0] - x0) / roi_w * roi_size
    lm[:, 1] = (lm[:, 1] - y0) / roi_h * roi_size

    return render_geometry_heatmap(lm, roi_size, roi_size, sigma=sigma, indices=indices)
