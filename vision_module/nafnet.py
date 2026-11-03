"""Frozen single-frame NAFNet inference used by labelled FER validation."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.baselines import NAFNetSingle
from .config import resolve_project_path
from .emotion import _rgb_uint8


class FrozenNAFNetRestorer:
    """Load the fixed NAFNet-SF checkpoint without exposing training code."""

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cuda") -> None:
        path = resolve_project_path(checkpoint)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"NAFNet checkpoint not found: {path}")
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA was requested but is unavailable; using CPU", RuntimeWarning)
            requested = torch.device("cpu")
        payload = torch.load(path, map_location=requested, weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload:
            raise TypeError("Unsupported NAFNet checkpoint format")
        fixed = payload.get("fixed_configuration", {})
        self.model = NAFNetSingle(
            width=int(fixed.get("width", 24)),
            blocks=tuple(fixed.get("blocks", (1, 1, 2, 2, 1, 1, 1))),
        ).to(requested)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval().requires_grad_(False)
        self.checkpoint_path = path
        self.device = requested

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def restore(self, frame) -> dict:
        rgb = _rgb_uint8(frame)
        h, w = rgb.shape[:2]
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        pad_h = (-h) % 8
        pad_w = (-w) % 8
        padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else tensor
        self._synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            restored = self.model(padded)[..., :h, :w]
        self._synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        output = (restored[0].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        return {"restored_frame": output, "latency_ms": latency_ms}
