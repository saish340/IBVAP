"""Base interface shared by every inference capability."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseAnalyzer(ABC):
    """An inference capability that processes single BGR frames.

    Concrete capabilities (detection, OCR, face recognition, ...) subclass
    this and implement :meth:`_load_model` + :meth:`analyze`. Models load
    lazily on first use, so the whole package stays importable even when a
    heavy dependency (TensorFlow, Paddle, ...) is not installed yet.

    Call :meth:`process` from threads: it serialises access to the model.
    """

    #: unique capability name, used by the registry in ``inference/__init__.py``
    name: str = "base"

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device
        self._model: Any = None
        self._loaded = False
        self._lock = threading.Lock()
        self.annotated_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------- life-cycle
    def ensure_loaded(self) -> None:
        """Load the model on first use."""
        if self._loaded:
            return
        self._model = self._load_model()
        self._loaded = True

    def warmup(self) -> None:
        """Pre-load the model (call at startup to avoid first-frame latency)."""
        self.ensure_loaded()

    def close(self) -> None:
        """Release the model. Override if the backend needs explicit cleanup."""
        self._model = None
        self._loaded = False

    # ---------------------------------------------------------------- analysis
    @abstractmethod
    def _load_model(self) -> Any:
        """Create and return the underlying model object."""

    @abstractmethod
    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run inference on a BGR frame and return a JSON-serialisable dict.

        The dict must include at least ``{"capability": self.name, ...}``.
        """

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """Thread-safe entry point around :meth:`analyze`."""
        with self._lock:
            return self.analyze(frame)
