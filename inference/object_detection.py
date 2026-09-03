"""Object detection capability backed by Ultralytics YOLO."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAnalyzer


class ObjectDetector(BaseAnalyzer):
    """Detect objects in frames with a YOLO model (default: ``yolov8n.pt``).

    Model weights are downloaded automatically by Ultralytics on first run
    and cached in the user's home directory.
    """

    name = "detection"

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.4,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device)
        self.model_name = model_name
        self.confidence = confidence

    def _load_model(self) -> Any:
        from ultralytics import YOLO  # heavy import kept lazy

        return YOLO(self.model_name)

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        self.ensure_loaded()
        results = self._model.predict(
            frame, conf=self.confidence, device=self.device, verbose=False
        )
        detections: List[Dict[str, Any]] = []
        for result in results:
            names = result.names
            for box in getattr(result, "boxes", None) or []:
                detections.append(
                    {
                        "class": names.get(int(box.cls), str(int(box.cls))),
                        "confidence": round(float(box.conf), 4),
                        "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                    }
                )
        return {"capability": self.name, "detections": detections, "count": len(detections)}
