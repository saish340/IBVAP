"""IBVAP inference capabilities.

Each capability lives in its own module and subclasses
:class:`~inference.base.BaseAnalyzer`. Models load lazily on first use, so
importing this package never triggers a heavy dependency or model download.

Built-in capabilities:

====================  ======================================================
Name                  Module / class
====================  ======================================================
``detection``         :class:`inference.object_detection.ObjectDetector`
``tracking``          :class:`inference.tracking.ObjectTracker`
``face_recognition``  :class:`inference.face_recognition.FaceRecognizer`
``ocr``               :class:`inference.ocr.OCRReader`
``pose``              :class:`inference.pose_estimation.PoseEstimator`
====================  ======================================================
"""

from typing import Any

from .base import BaseAnalyzer
from .face_recognition import FaceRecognizer
from .object_detection import ObjectDetector
from .ocr import OCRReader
from .pose_estimation import PoseEstimator
from .tracking import ObjectTracker

#: Maps capability names to their analyzer classes.
ANALYZERS: dict[str, type[BaseAnalyzer]] = {
    ObjectDetector.name: ObjectDetector,
    FaceRecognizer.name: FaceRecognizer,
    OCRReader.name: OCRReader,
    PoseEstimator.name: PoseEstimator,
    ObjectTracker.name: ObjectTracker,
}

__all__ = [
    "ANALYZERS",
    "BaseAnalyzer",
    "FaceRecognizer",
    "ObjectDetector",
    "OCRReader",
    "ObjectTracker",
    "PoseEstimator",
    "get_analyzer",
    "list_capabilities",
]


def list_capabilities() -> list[str]:
    """Return the names of all registered capabilities."""
    return sorted(set(ANALYZERS) | {"face_verification", "anpr"})


def get_analyzer(name: str, **kwargs: Any) -> BaseAnalyzer:
    """Build an analyzer by capability name.

    Raises ``ValueError`` for unknown names, listing what is available.
    """
    try:
        analyzer_cls = ANALYZERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown capability '{name}'. Available: {', '.join(list_capabilities())}"
        ) from None
    return analyzer_cls(**kwargs)
