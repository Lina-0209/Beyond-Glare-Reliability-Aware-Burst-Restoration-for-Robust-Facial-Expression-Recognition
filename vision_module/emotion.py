"""Frozen pretrained facial emotion recognition adapter."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from PIL import Image

CANONICAL_LABELS = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}


def _rgb_uint8(face: Any) -> np.ndarray:
    if isinstance(face, Image.Image):
        return np.asarray(face.convert("RGB"))
    if torch.is_tensor(face):
        value = face.detach().cpu()
        if value.ndim != 3:
            raise ValueError(f"FER input tensor must be 3D, got {tuple(value.shape)}")
        if value.shape[0] == 3:
            value = value.permute(1, 2, 0)
        face = value.numpy()
    array = np.asarray(face)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"FER expects an RGB HxWx3 image, got {array.shape}")
    if array.dtype == np.uint8:
        return array if array.flags.c_contiguous else np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating) and float(array.max(initial=0)) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


class EmotionRecognizer:
    """Use EmotiEffLib's pretrained AffectNet seven-class model."""

    def __init__(
        self,
        model_name: str = "enet_b2_7",
        device: str | torch.device = "cuda",
        *,
        engine: str = "onnx",
        backend=None,
    ) -> None:
        self.device = torch.device(
            "cpu" if str(device).startswith("cuda") and not torch.cuda.is_available() else device
        )
        if backend is None:
            try:
                from emotiefflib.facial_analysis import (
                    EmotiEffLibRecognizerOnnx,
                    EmotiEffLibRecognizerTorch,
                )
            except ImportError as error:
                raise ImportError(
                    "EmotionRecognizer requires emotiefflib==1.1.1. "
                    "Install it with: pip install --no-deps emotiefflib==1.1.1"
                ) from error
            if engine == "onnx":
                backend = EmotiEffLibRecognizerOnnx(model_name=model_name)
                self.device = torch.device("cpu")
            elif engine == "torch":
                backend = EmotiEffLibRecognizerTorch(
                    model_name=model_name, device=str(self.device)
                )
            else:
                raise ValueError(f"Unsupported FER engine: {engine}")
        self.backend = backend
        if hasattr(self.backend, "model"):
            self.backend.model.eval()
            self.backend.model.requires_grad_(False)

    def predict(self, face) -> dict:
        """Return canonical seven-class probabilities for one RGB face image."""
        rgb = _rgb_uint8(face)
        started = time.perf_counter()
        with torch.inference_mode():
            labels, scores = self.backend.predict_emotions(rgb, logits=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim == 2:
            values = values[0]
        class_names = getattr(self.backend, "idx_to_emotion_class", {})
        if len(class_names) != len(values):
            raise RuntimeError("FER backend returned an unexpected class count")
        values = np.clip(values, 0.0, None)
        total = float(values.sum())
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("FER backend returned invalid probabilities")
        values /= total
        probabilities = {
            CANONICAL_LABELS.get(str(class_names[index]).lower(), str(class_names[index]).lower()):
            float(values[index])
            for index in range(len(values))
        }
        predicted_native = str(labels[0] if isinstance(labels, (list, tuple)) else labels).lower()
        emotion = CANONICAL_LABELS.get(predicted_native, predicted_native)
        confidence = float(probabilities[emotion])
        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probabilities,
            "latency_ms": latency_ms,
        }
