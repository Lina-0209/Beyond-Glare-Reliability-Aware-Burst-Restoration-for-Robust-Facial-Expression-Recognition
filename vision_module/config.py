"""Configuration helpers for the vision module."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG: dict[str, Any] = {
    "device": "cuda",
    "berr": {
        "checkpoint": None,
        "config": None,
        "num_frames": 4,
        "image_size": 256,
        "roi_size": 128,
        "face_landmarker": "assets/face_landmarker.task",
        "glare_presence_threshold": 0.05,
        "reuse_anchor_landmarks": True,
        "geometry_cache_enabled": False,
        "geometry_cache_max_age": 4,
        "geometry_cache_motion_threshold": 0.08,
    },
    "fer": {"model": "enet_b2_7", "engine": "onnx"},
    "vision": {
        "emotion_confidence_threshold": 0.60,
        "confidence_policy": "artifacts/fer_confidence_policy_v1.json",
        "emotion_margin_threshold": 0.0,
        "restoration_confidence_drop_threshold": -0.05,
        "compare_pre_restore_fer": True,
        "pre_restore_fer_interval": 4,
        "parallel_gate_fer": True,
        "mode": "always_restore",
    },
    "glare_gate": {"enabled": False, "model_path": "checkpoints/fast_glare_gate.pkl"},
}


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_vision_config(config: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load defaults plus a YAML path or mapping override."""
    result = deepcopy(DEFAULT_CONFIG)
    if config is None:
        return result
    if isinstance(config, Mapping):
        return _deep_update(result, config)
    path = Path(config)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, Mapping):
        raise TypeError(f"Vision config must contain a mapping: {path}")
    return _deep_update(result, values)


def resolve_project_path(path: str | Path | None) -> Path | None:
    """Resolve repository-relative configuration paths."""
    if path is None:
        return None
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
