"""Adapter for the frozen production EmotiEffLib backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base import FERBackend
from .schemas import BackendResult, normalize_probabilities


class EmotiEffBackend(FERBackend):
    name = "EmotiEffLib/enet_b2_7"

    def __init__(self, device: str = "cpu", engine: str = "onnx") -> None:
        from vision_module.emotion import EmotionRecognizer
        self.recognizer = EmotionRecognizer(model_name="enet_b2_7", device=device, engine=engine)

    def predict(self, image_path: str | Path) -> BackendResult:
        with Image.open(image_path) as image:
            result = self.recognizer.predict(np.asarray(image.convert("RGB"), dtype=np.uint8))
        return BackendResult(
            backend=self.name,
            native_prediction=result["emotion"],
            emotion=result["emotion"],
            emotion_probs=result["probabilities"],
            emotion_confidence=float(result["confidence"]),
            latency_ms=float(result["latency_ms"]),
        )

    def predict_batch(self, image_paths: list[str | Path]) -> list[BackendResult]:
        import time
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        started = time.perf_counter()
        labels, scores = self.recognizer.backend.predict_emotions(images, logits=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim == 1:
            scores = scores[None, :]
        class_names = getattr(self.recognizer.backend, "idx_to_emotion_class", {})
        results = []
        for index, label in enumerate(labels):
            values = np.clip(scores[index], 0.0, None)
            probs = normalize_probabilities({
                self.recognizer.backend.idx_to_emotion_class[i].lower(): float(values[i])
                for i in range(len(values))
            })
            canonical = self.recognizer.predict(images[index])["emotion"] if not probs else str(label).lower()
            canonical = self.recognizer.backend.idx_to_emotion_class[int(np.argmax(values))].lower()
            from vision_module.emotion import CANONICAL_LABELS
            canonical = CANONICAL_LABELS.get(canonical, canonical)
            probs = {CANONICAL_LABELS.get(key, key): value for key, value in probs.items()}
            results.append(BackendResult(
                backend=self.name, native_prediction=str(label).lower(), emotion=canonical,
                emotion_probs=probs, emotion_confidence=float(probs.get(canonical, 0.0)),
                latency_ms=elapsed_ms / max(len(image_paths), 1),
            ))
        return results
