"""Multi-object tracking capability backed by Ultralytics YOLO tracking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAnalyzer

TRACK_CLASSES = {"person", "car", "truck", "motorcycle", "bus"}


class ObjectTracker(BaseAnalyzer):
    """Track objects across frames (YOLO + ByteTrack/BoT-SORT).

    ``persist=True`` keeps track state between calls, so feed frames from a
    single stream to a single tracker instance.
    """

    name = "tracking"

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker: str = "bytetrack.yaml",
        confidence: float = 0.4,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device)
        self.model_name = model_name
        self.tracker = tracker
        self.confidence = confidence

    def _load_model(self) -> Any:
        from ultralytics import YOLO  # heavy import kept lazy

        return YOLO(self.model_name)

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        self.ensure_loaded()
        results = self._model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            classes=[0, 2, 3, 5, 7],
            device=self.device,
            verbose=False,
        )
        tracks: List[Dict[str, Any]] = []
        for result in results:
            self.annotated_frame = result.plot()
            names = result.names
            for box in getattr(result, "boxes", None) or []:
                track_id = box.id
                class_name = names.get(int(box.cls), str(int(box.cls)))
                if class_name not in TRACK_CLASSES:
                    continue
                tracks.append(
                    {
                        "track_id": int(track_id) if track_id is not None else None,
                        "class": class_name,
                        "confidence": round(float(box.conf), 4),
                        "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                    }
                )
        return {"capability": self.name, "tracks": tracks, "count": len(tracks)}
