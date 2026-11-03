"""Backend-independent FER result schemas and optional adapters."""

from .schemas import CANONICAL_LABELS, BackendResult, normalize_probabilities

__all__ = ["CANONICAL_LABELS", "BackendResult", "normalize_probabilities"]
