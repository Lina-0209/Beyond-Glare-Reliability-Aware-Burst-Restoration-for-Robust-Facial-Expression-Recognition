"""Minimal adapter protocol used by benchmark runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .schemas import BackendResult


class FERBackend(ABC):
    name: str

    @abstractmethod
    def predict(self, image_path: str | Path) -> BackendResult:
        raise NotImplementedError

    def predict_batch(self, image_paths: list[str | Path]) -> list[BackendResult]:
        return [self.predict(path) for path in image_paths]
