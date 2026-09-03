"""Human pose estimation capability backed by MediaPipe Pose."""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from .base import BaseAnalyzer

# Landmark indices for the 33-point MediaPipe body model.
_KEYPOINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


class PoseEstimator(BaseAnalyzer):
    """Estimate human body pose (33 landmarks) with MediaPipe Pose.

    Note: MediaPipe Pose is single-person. For multi-person pose, run object
    detection first and crop per person before calling this analyzer.
    """

    name = "pose"

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        static_image_mode: bool = False,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device)
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.static_image_mode = static_image_mode

    def _load_model(self) -> Any:
        import mediapipe as mp  # kept lazy; large dependency

        self._pose_module = mp.solutions.pose
        return self._pose_module.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
        )

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        self.ensure_loaded()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._model.process(rgb)
        if result.pose_landmarks is None:
            return {"capability": self.name, "poses": [], "count": 0}

        landmarks = [
            {
                "x": round(float(p.x), 5),
                "y": round(float(p.y), 5),
                "z": round(float(p.z), 5),
                "visibility": round(float(p.visibility), 4),
            }
            for p in result.pose_landmarks.landmark
        ]
        keypoints = {
            key: landmarks[index] for key, index in _KEYPOINTS.items() if index < len(landmarks)
        }
        return {
            "capability": self.name,
            "poses": [{"landmarks": landmarks, "keypoints": keypoints}],
            "count": 1,
        }

    def close(self) -> None:
        if self._model is not None:
            try:
                self._model.close()
            except Exception:
                pass
        super().close()
