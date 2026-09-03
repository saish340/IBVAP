"""Face detection + recognition capability backed by DeepFace."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAnalyzer


class FaceRecognizer(BaseAnalyzer):
    """Find faces in frames and compute embeddings with DeepFace.

    DeepFace downloads its model weights on first use. To recognise known
    people, compare the returned embeddings against enrolled ones (or call
    ``DeepFace.find`` in your own service) — this module stays stateless.
    """

    name = "face_recognition"

    def __init__(
        self,
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        enforce_detection: bool = False,
        include_embeddings: bool = False,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device)
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        self.include_embeddings = include_embeddings

    def _load_model(self) -> Any:
        from deepface import DeepFace  # pulls in TensorFlow; kept lazy

        return DeepFace

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        self.ensure_loaded()
        faces = self._model.represent(
            img_path=frame,  # DeepFace accepts BGR numpy arrays
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            enforce_detection=self.enforce_detection,
        )
        results: List[Dict[str, Any]] = []
        for face in faces:
            area = face.get("facial_area", {})
            entry: Dict[str, Any] = {
                "confidence": round(float(face.get("face_confidence", 0.0)), 4),
                "bbox": [
                    float(area.get("x", 0)),
                    float(area.get("y", 0)),
                    float(area.get("x", 0)) + float(area.get("w", 0)),
                    float(area.get("y", 0)) + float(area.get("h", 0)),
                ],
            }
            if self.include_embeddings:
                embedding = face.get("embedding")
                entry["embedding"] = embedding
                entry["embedding_dim"] = len(embedding or [])
            results.append(entry)
        return {"capability": self.name, "faces": results, "count": len(results)}
