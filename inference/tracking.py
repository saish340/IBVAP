"""Multi-object tracking capability backed by Ultralytics YOLO tracking."""

from __future__ import annotations

import os
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
        imgsz: int = 640,
    ) -> None:
        super().__init__(device=device)
        self.model_name = model_name
        self.tracker = tracker
        self.confidence = confidence
        # Lower image size = faster CPU inference.  640 is the YOLOv8 default
        # sweet spot for person/vehicle detection; do not drop below ~480.
        self.imgsz = imgsz

    def _load_model(self) -> Any:
        from ultralytics import YOLO  # heavy import kept lazy

        model = YOLO(self.model_name)
        # PyTorch defaults to one thread per core (16 on this i7-13700HX).
        # For a small model like yolov8n, 16 threads oversubscribe the CPU and
        # collide with the ANPR ONNX thread pool, inflating latency 5-15x.
        # A modest cap keeps single-inference latency lowest for this app.
        try:
            import torch

            torch.set_num_threads(int(os.getenv("YOLO_THREADS", "4")))
        except Exception:
            pass
        return model

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        self.ensure_loaded()
        results = self._model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            imgsz=self.imgsz,
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
