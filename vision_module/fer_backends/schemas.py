"""Shared output schema for isolated FER backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

CANONICAL_LABELS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")


def normalize_probabilities(values: dict[str, float]) -> dict[str, float]:
    clean = {str(key): float(value) for key, value in values.items()
             if np.isfinite(float(value)) and float(value) >= 0.0}
    total = float(sum(clean.values()))
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in clean.items()}


@dataclass
class BackendResult:
    backend: str
    native_prediction: str | None
    emotion: str | None
    emotion_probs: dict[str, float] = field(default_factory=dict)
    emotion_confidence: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
