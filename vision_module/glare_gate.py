"""Fast anchor-frame glare routing for conditional BERR execution."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import torch

from datasets.face_geometry import compute_roi_bbox_from_landmarks, extract_landmarks
from .preprocessing import _as_rgb_array


def legacy_glare_features(rgb: np.ndarray, bbox) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    roi = rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
    if roi.size == 0:
        raise ValueError("Empty glare-gate ROI")
    lum = 0.299 * roi[..., 0] + 0.587 * roi[..., 1] + 0.114 * roi[..., 2]
    saturation = roi.max(2) - roi.min(2)
    gx = np.abs(np.diff(lum, axis=1)).mean() if lum.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(lum, axis=0)).mean() if lum.shape[0] > 1 else 0.0
    return np.array([
        *np.percentile(lum, [50, 75, 90, 95, 99]), lum.std(), gx + gy,
        np.mean(lum > 0.75), np.mean(lum > 0.9), np.mean((lum > 0.75) & (saturation < 0.16)),
        saturation.mean(), saturation.std(),
    ], dtype=np.float32)


def glare_features(rgb: np.ndarray, bbox) -> np.ndarray:
    """Static eye-ROI features for the calibrated three-way glare router."""
    import cv2

    x0, y0, x1, y1 = [int(v) for v in bbox]
    roi = rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
    if roi.size == 0:
        raise ValueError("Empty glare-gate ROI")
    lum = 0.299 * roi[..., 0] + 0.587 * roi[..., 1] + 0.114 * roi[..., 2]
    saturation = roi.max(2) - roi.min(2)
    value = roi.max(2)
    gx = np.abs(np.diff(lum, axis=1)) if lum.shape[1] > 1 else np.zeros_like(lum)
    gy = np.abs(np.diff(lum, axis=0)) if lum.shape[0] > 1 else np.zeros_like(lum)
    gradient = np.concatenate([gx.ravel(), gy.ravel()])
    laplacian = np.abs(cv2.Laplacian(lum, cv2.CV_32F)).ravel()
    features = [
        *np.percentile(lum, [5, 10, 25, 50, 75, 85, 90, 95, 97, 99]),
        lum.mean(), lum.std(),
        *np.percentile(saturation, [10, 25, 50, 75, 90]),
        saturation.mean(), saturation.std(),
        *np.percentile(value, [50, 75, 90, 95, 99]),
        *[np.mean(lum > threshold) for threshold in (.65, .75, .85, .90, .95)],
        *[
            np.mean((lum > threshold) & (saturation < limit))
            for threshold, limit in ((.65, .20), (.75, .20), (.85, .20), (.75, .12), (.85, .12))
        ],
        np.mean(value >= .98), np.mean(roi.min(2) >= .98),
        gradient.mean(), gradient.std(), *np.percentile(gradient, [75, 90, 95]),
        laplacian.mean(), laplacian.std(), *np.percentile(laplacian, [75, 90, 95]),
    ]
    middle = max(1, lum.shape[1] // 2)
    left, right = lum[:, :middle], lum[:, middle:]
    features.extend([
        abs(float(left.mean() - right.mean())),
        abs(float(np.percentile(left, 95) - np.percentile(right, 95))),
        abs(float(np.mean(left > .8) - np.mean(right > .8))),
    ])
    block_scores = []
    for y_indices in np.array_split(np.arange(lum.shape[0]), 2):
        for x_indices in np.array_split(np.arange(lum.shape[1]), 2):
            block_scores.append(float(np.percentile(lum[np.ix_(y_indices, x_indices)], 95)))
    features.extend(sorted(block_scores))
    return np.asarray(features, dtype=np.float32)


class FastGlareGate:
    def __init__(self, artifact, *, face_landmarker_path=None):
        artifact = Path(artifact)
        if artifact.suffix == ".pkl":
            with artifact.open("rb") as handle: payload = pickle.load(handle)
        else:
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
        self.model_type = payload.get("model_type", "linear")
        inferred_version = (
            "legacy_v1"
            if self.model_type == "linear" and np.asarray(payload.get("weight", [])).size == 12
            else "enhanced_static_v2"
        )
        self.feature_version = payload.get("feature_version", inferred_version)
        self.model = payload.get("model")
        if self.model_type == "linear":
            self.mean = np.asarray(payload["feature_mean"], dtype=np.float32)
            self.scale = np.asarray(payload["feature_scale"], dtype=np.float32)
            self.weight = np.asarray(payload["weight"], dtype=np.float32)
            self.bias = float(payload["bias"])
        self.threshold = float(payload.get("threshold", payload.get("restore_threshold", .5)))
        self.direct_threshold = float(payload.get("direct_threshold", self.threshold))
        self.restore_threshold = float(payload.get("restore_threshold", self.threshold))
        if self.direct_threshold > self.restore_threshold:
            raise ValueError("Glare-gate direct threshold must not exceed restore threshold")
        self.face_landmarker_path = face_landmarker_path

    def predict(self, frame, *, landmarks=None, roi_bbox=None):
        rgb = _as_rgb_array(frame)
        points = np.asarray(landmarks if landmarks is not None else extract_landmarks(rgb, self.face_landmarker_path), dtype=np.float32)
        if points.ndim != 2:
            raise ValueError("No face detected for glare gate")
        bbox = compute_roi_bbox_from_landmarks(points, *rgb.shape[:2]) if roi_bbox is None else roi_bbox
        if torch.is_tensor(bbox):
            bbox = bbox.detach().cpu().reshape(-1).tolist()
        feature_fn = legacy_glare_features if self.feature_version == "legacy_v1" else glare_features
        feature = feature_fn(rgb, bbox)
        if self.model_type == "linear":
            z = (feature - self.mean) / self.scale
            probability = float(1.0 / (1.0 + np.exp(-float(z @ self.weight + self.bias))))
        elif self.model_type == "hist_gradient_boosting":
            probability = float(self.model.predict_proba(feature[None])[0, 1])
        else:
            raise ValueError(f"Unsupported glare-gate model type: {self.model_type}")
        if probability <= self.direct_threshold:
            route = "direct"
        elif probability >= self.restore_threshold:
            route = "restore"
        else:
            route = "abstain"
        return {"probability": probability, "route": route, "detected": route == "restore",
                "landmarks": points, "roi_bbox": [int(v) for v in bbox], "rgb": rgb}
