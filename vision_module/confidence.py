"""Calibrated confidence, top-2, and abstention policy for frozen FER output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


class ConfidencePolicy:
    """Apply validation-calibrated thresholds without changing FER weights."""

    def __init__(
        self,
        artifact: str | Path | None = None,
        *,
        default_threshold: float = 0.60,
        margin_threshold: float = 0.0,
        restoration_drop_threshold: float = -0.05,
    ) -> None:
        payload: dict = {}
        self.artifact_path: Path | None = None
        if artifact is not None:
            path = Path(artifact)
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.artifact_path = path
        self.default_threshold = float(payload.get("default_confidence_threshold", default_threshold))
        self.margin_threshold = float(payload.get("margin_threshold", margin_threshold))
        self.restoration_drop_threshold = float(
            payload.get("restoration_confidence_drop_threshold", restoration_drop_threshold)
        )
        self.class_thresholds = {
            str(label): float(value)
            for label, value in payload.get("class_thresholds", {}).items()
        }
        self.metadata = payload.get("metadata", {})

    def assess(self, prediction: Mapping) -> dict:
        probabilities = {
            str(label): float(value)
            for label, value in prediction.get("probabilities", {}).items()
        }
        ranked = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
        if not ranked:
            emotion = str(prediction.get("emotion", "unknown"))
            ranked = [(emotion, float(prediction.get("confidence", 0.0)))]
        top2 = [
            {"emotion": label, "probability": probability}
            for label, probability in ranked[:2]
        ]
        confidence = float(prediction.get("confidence", ranked[0][1]))
        margin = float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else confidence
        raw_emotion = str(prediction.get("emotion", ranked[0][0]))
        threshold = float(self.class_thresholds.get(raw_emotion, self.default_threshold))
        reasons = []
        if confidence < threshold:
            reasons.append("below_class_confidence_threshold")
        if margin < self.margin_threshold:
            reasons.append("below_top2_margin_threshold")
        return {
            "raw_emotion": raw_emotion,
            "top2_emotions": top2,
            "emotion_margin": margin,
            "confidence_threshold": threshold,
            "margin_threshold": self.margin_threshold,
            "low_confidence": bool(reasons),
            "uncertainty_reasons": reasons,
        }

    def restoration_drop(self, before: Mapping | None, after: Mapping) -> tuple[float | None, bool | None]:
        if before is None:
            return None, None
        delta = float(after.get("confidence", 0.0)) - float(before.get("confidence", 0.0))
        return delta, delta < self.restoration_drop_threshold

