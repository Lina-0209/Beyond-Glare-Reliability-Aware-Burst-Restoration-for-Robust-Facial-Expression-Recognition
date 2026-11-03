"""Official POSTER V2 AffectNet-7 adapter.

The upstream model constructor contains two Windows-only absolute paths for its
frozen landmark/IR pretraining weights.  The adapter remaps only those paths at
load time; the checked-out upstream source and checkpoint remain unchanged.
"""

from __future__ import annotations

import collections
import importlib
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .base import FERBackend
from .schemas import BackendResult, normalize_probabilities


POSTER_LABELS = ("neutral", "happy", "sad", "surprise", "fear", "disgust", "angry")


class RecorderMeter1:  # noqa: D101 - compatibility classes for upstream pickle
    pass


class RecorderMeter:  # noqa: D101 - compatibility classes for upstream pickle
    pass


class PosterV2Backend(FERBackend):
    name = "POSTER_V2/AffectNet-7"

    def __init__(self, repo_root: str | Path, checkpoint: str | Path,
                 device: str = "cuda") -> None:
        self.repo_root = Path(repo_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        # The host project also has a top-level ``models`` package, which may
        # already be loaded. Load upstream modules under a private alias so
        # their relative imports cannot collide with the host package.
        package_name = "poster_official_models"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(self.repo_root / "models")]
            package.__package__ = package_name
            sys.modules[package_name] = package
        model_module = importlib.import_module(f"{package_name}.PosterV2_7cls")
        pyramid_trans_expr2 = model_module.pyramid_trans_expr2
        with self._upstream_path_remap():
            self.model = pyramid_trans_expr2(img_size=224, num_classes=7).to(self.device)
            payload = self._load_checkpoint(self.checkpoint)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        clean_state = collections.OrderedDict()
        for key, value in state.items():
            clean_state[key[7:] if key.startswith("module.") else key] = value
        missing, unexpected = self.model.load_state_dict(clean_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"POSTER checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        self.model.eval().requires_grad_(False)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _load_checkpoint(path: Path):
        import __main__
        # The official checkpoint was saved from ``main.py`` and pickles two
        # training-only recorder objects as ``__main__`` globals. They are not
        # needed for inference. Only load trusted upstream checkpoints: this is pickle.
        __main__.RecorderMeter1 = RecorderMeter1
        __main__.RecorderMeter = RecorderMeter
        return torch.load(path, map_location="cpu", weights_only=False)

    def _upstream_path_remap(self):
        original_load = torch.load
        ir_path = str(self.repo_root / "models/pretrain/ir50.pth")
        mobile_path = str(self.repo_root / "models/pretrain/mobilefacenet_model_best.pth.tar")

        def remapped_load(path, *args, **kwargs):
            value = str(path)
            if value.endswith("ir50.pth") or "\\ir50.pth" in value:
                path = ir_path
            elif value.endswith("mobilefacenet_model_best.pth.tar") or "\\mobilefacenet_model_best.pth.tar" in value:
                path = mobile_path
            return original_load(path, *args, **kwargs)

        return patch.object(torch, "load", remapped_load)

    def predict(self, image_path: str | Path) -> BackendResult:
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        index = int(np.argmax(probs))
        native = POSTER_LABELS[index]
        probability_map = normalize_probabilities({label: float(probs[i]) for i, label in enumerate(POSTER_LABELS)})
        return BackendResult(
            backend=self.name,
            native_prediction=native,
            emotion=native,
            emotion_probs=probability_map,
            emotion_confidence=float(probs[index]),
            extra={"native_labels": list(POSTER_LABELS), "output_index": index},
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def predict_batch(self, image_paths: list[str | Path]) -> list[BackendResult]:
        tensors = []
        for path in image_paths:
            with Image.open(path) as image:
                tensors.append(self.transform(image.convert("RGB")))
        batch = torch.stack(tensors, dim=0).to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        per_image_ms = (time.perf_counter() - started) * 1000.0 / max(len(image_paths), 1)
        results = []
        for values in probabilities:
            index = int(np.argmax(values))
            native = POSTER_LABELS[index]
            probs = normalize_probabilities({label: float(values[i]) for i, label in enumerate(POSTER_LABELS)})
            results.append(BackendResult(
                backend=self.name, native_prediction=native, emotion=native,
                emotion_probs=probs, emotion_confidence=float(values[index]),
                extra={"native_labels": list(POSTER_LABELS), "output_index": index},
                latency_ms=per_image_ms,
            ))
        return results
