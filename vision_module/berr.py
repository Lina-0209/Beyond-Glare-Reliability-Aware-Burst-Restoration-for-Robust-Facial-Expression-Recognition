"""High-level, training-independent BERR inference wrapper."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import yaml

from models.berr import BurstEyeglassReflectionRemoval
from scripts.train_berr import load_model_weights

from .config import resolve_project_path
from .preprocessing import PreparedBurst, _as_rgb_array, prepare_burst

FROZEN_MODEL_CONFIG = {
    "roi_size": 128,
    "width": 32,
    "encoder_blocks": [2, 2, 4],
    "decoder_blocks": [2, 2],
    "reliability_temperature": 0.15,
    "use_offset_head": False,
    "max_residual_flow": 4.0,
    "feather_pixels": 3,
}


def _device(value: str | torch.device) -> torch.device:
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA was requested but is unavailable; using CPU", RuntimeWarning)
        return torch.device("cpu")
    return requested


def _read_model_config(config) -> tuple[dict, dict]:
    if config is None:
        return {}, {}
    if isinstance(config, Mapping):
        loaded = dict(config)
    else:
        path = resolve_project_path(config)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    return dict(loaded.get("model", loaded)), dict(loaded.get("dataset", {}))


class BERRRestorer:
    """Hide BERR model internals behind a four-frame restoration API."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "cuda",
        config=None,
        *,
        image_size: int | None = None,
        num_frames: int = 4,
        face_landmarker_path: str | Path | None = None,
        glare_presence_threshold: float = 0.05,
        reuse_anchor_landmarks: bool = True,
        geometry_cache_enabled: bool = False,
        geometry_cache_max_age: int = 4,
        geometry_cache_motion_threshold: float = 0.08,
    ) -> None:
        checkpoint_path = resolve_project_path(checkpoint)
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError(f"BERR checkpoint not found: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path
        self.device = _device(device)
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("Unsupported BERR checkpoint format")

        explicit_model, explicit_dataset = _read_model_config(config)
        checkpoint_cfg = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
        model_cfg = dict(FROZEN_MODEL_CONFIG)
        model_cfg.update(checkpoint_cfg.get("model", {}))
        model_cfg.update(explicit_model)
        self.model = BurstEyeglassReflectionRemoval(**model_cfg).to(self.device)
        load_model_weights(self.model, payload)
        self.model.eval()
        self.model.requires_grad_(False)

        checkpoint_dataset = checkpoint_cfg.get("dataset", {})
        self.image_size = int(
            image_size
            or explicit_dataset.get("image_size")
            or checkpoint_dataset.get("image_size")
            or 256
        )
        self.roi_size = int(model_cfg["roi_size"])
        self.num_frames = int(num_frames)
        self.face_landmarker_path = resolve_project_path(face_landmarker_path)
        self.glare_presence_threshold = float(glare_presence_threshold)
        self.reuse_anchor_landmarks = bool(reuse_anchor_landmarks)
        self.geometry_cache_enabled = bool(geometry_cache_enabled)
        self.geometry_cache_max_age = max(1, int(geometry_cache_max_age))
        self.geometry_cache_motion_threshold = float(geometry_cache_motion_threshold)
        self._geometry_cache: dict | None = None
        self._device_input_buffers: dict[str, torch.Tensor] = {}

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _thumbnail(frame: np.ndarray, size: int = 24) -> np.ndarray:
        """Return a cheap normalized grayscale thumbnail for cache invalidation."""
        height, width = frame.shape[:2]
        y = np.linspace(0, height - 1, min(size, height)).round().astype(np.intp)
        x = np.linspace(0, width - 1, min(size, width)).round().astype(np.intp)
        sampled = frame[y[:, None], x[None, :]].astype(np.float32)
        return (
            0.299 * sampled[..., 0]
            + 0.587 * sampled[..., 1]
            + 0.114 * sampled[..., 2]
        ) / 255.0

    def reset_geometry_cache(self) -> None:
        """Forget stream-local landmark/ROI state (for a cut or a new subject)."""
        self._geometry_cache = None

    def _cached_geometry(self, anchor: np.ndarray) -> tuple[np.ndarray | None, list[int] | None, bool]:
        if not self.geometry_cache_enabled or self._geometry_cache is None:
            return None, None, False
        cached = self._geometry_cache
        if tuple(anchor.shape[:2]) != tuple(cached["shape"]):
            return None, None, False
        if int(cached["age"]) >= self.geometry_cache_max_age:
            return None, None, False
        difference = float(np.mean(np.abs(self._thumbnail(anchor) - cached["thumbnail"])))
        if difference > self.geometry_cache_motion_threshold:
            return None, None, False
        cached["age"] = int(cached["age"]) + 1
        return cached["landmarks"].copy(), list(cached["roi_bbox"]), True

    def _update_geometry_cache(self, prepared: PreparedBurst, anchor: np.ndarray) -> None:
        if not self.geometry_cache_enabled:
            return
        original_h, original_w = prepared.original_shapes[0]
        points = prepared.landmarks[0, 0].detach().cpu().numpy().copy()
        points[:, 0] *= original_w / float(self.image_size)
        points[:, 1] *= original_h / float(self.image_size)
        bbox = prepared.roi_bbox[0].detach().cpu().numpy().astype(np.float32)
        bbox[[0, 2]] *= original_w / float(self.image_size)
        bbox[[1, 3]] *= original_h / float(self.image_size)
        self._geometry_cache = {
            "shape": (original_h, original_w),
            "thumbnail": self._thumbnail(anchor),
            "landmarks": points,
            "roi_bbox": bbox.round().astype(int).tolist(),
            "age": 0,
        }

    def _move_inputs(self, prepared: PreparedBurst) -> dict[str, torch.Tensor]:
        host_inputs = {
            "burst": prepared.burst,
            "landmarks": prepared.landmarks,
            "roi_bbox": prepared.roi_bbox,
            "geometry_heatmap_roi": prepared.geometry_heatmap_roi,
        }
        if self.device.type != "cuda":
            return {name: value.to(self.device) for name, value in host_inputs.items()}
        moved = {}
        for name, value in host_inputs.items():
            buffer = self._device_input_buffers.get(name)
            if buffer is None or buffer.shape != value.shape or buffer.dtype != value.dtype:
                buffer = torch.empty(value.shape, dtype=value.dtype, device=self.device)
                self._device_input_buffers[name] = buffer
            buffer.copy_(value, non_blocking=False)
            moved[name] = buffer
        return moved

    def restore(self, frames, landmarks=None, roi_bbox=None, anchor_landmarks=None) -> dict:
        """Restore frame 0 and return RGB output plus BERR degradation estimates."""
        started = time.perf_counter()
        anchor = _as_rgb_array(frames[0])
        geometry_cache_hit = False
        if landmarks is None and anchor_landmarks is None and roi_bbox is None:
            anchor_landmarks, roi_bbox, geometry_cache_hit = self._cached_geometry(anchor)
        prepared: PreparedBurst = prepare_burst(
            frames,
            landmarks=landmarks,
            roi_bbox=roi_bbox,
            expected_frames=self.num_frames,
            image_size=self.image_size,
            roi_size=self.roi_size,
            face_landmarker_path=self.face_landmarker_path,
            anchor_landmarks=anchor_landmarks,
            reuse_anchor_landmarks=self.reuse_anchor_landmarks,
        )
        if not geometry_cache_hit:
            self._update_geometry_cache(prepared, anchor)
        preprocess_ms = (time.perf_counter() - started) * 1000.0
        inputs = self._move_inputs(prepared)
        self._synchronize()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(
                inputs["burst"],
                inputs["landmarks"],
                inputs["roi_bbox"],
                inputs["geometry_heatmap_roi"],
            )
        self._synchronize()
        berr_ms = (time.perf_counter() - inference_started) * 1000.0

        restored_tensor = outputs["final"][0].detach().cpu().clamp(0, 1)
        restored_rgb = (
            restored_tensor.permute(1, 2, 0).numpy() * 255.0
        ).round().astype(np.uint8)
        degradation = outputs["degradation"][0, :, 0].detach().cpu().numpy()
        glare_score = float(np.clip(degradation[0].mean(), 0.0, 1.0))
        return {
            "restored_frame": restored_rgb,
            "restored_tensor": restored_tensor,
            "anchor_frame": prepared.rgb_frames[0],
            "glare_score": glare_score,
            "glare_removed": glare_score >= self.glare_presence_threshold,
            "degradation_maps": degradation,
            "roi_bbox": prepared.roi_bbox[0].tolist(),
            "geometry_cache_hit": geometry_cache_hit,
            "landmark_strategy": "stream_cache" if geometry_cache_hit else (
                "anchor_reuse" if self.reuse_anchor_landmarks else "per_frame"
            ),
            "latency_ms": {
                "preprocessing": preprocess_ms,
                "berr": berr_ms,
            },
        }
