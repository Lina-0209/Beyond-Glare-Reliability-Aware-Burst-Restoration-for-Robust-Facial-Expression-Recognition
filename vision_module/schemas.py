"""Typed, JSON-serializable output schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class VisionState:
    timestamp: str
    emotion: str
    emotion_probs: dict[str, float]
    emotion_confidence: float
    glare_score: float
    glare_removed: bool
    vision_confidence: float
    low_confidence: bool
    processing_time_ms: float
    status: str
    restoration_mode: str = "always_restore"
    routing_decision: str | None = None
    berr_executed: bool = True
    fast_glare_probability: float | None = None
    fast_glare_detected: bool | None = None
    raw_emotion: str | None = None
    top2_emotions: list[dict[str, float | str]] = field(default_factory=list)
    emotion_margin: float | None = None
    confidence_threshold: float | None = None
    margin_threshold: float | None = None
    uncertainty_reasons: list[str] = field(default_factory=list)
    pre_restoration_emotion: str | None = None
    pre_restoration_confidence: float | None = None
    restoration_confidence_delta: float | None = None
    restoration_confidence_drop: bool | None = None
    pre_restoration_comparison_performed: bool = False
    speculative_fer_started: bool = False
    geometry_cache_hit: bool | None = None
    landmark_strategy: str | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return only standard JSON-compatible Python values."""
        return asdict(self)
