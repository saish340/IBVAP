"""Human pose estimation capability backed by MediaPipe Pose."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .base import BaseAnalyzer

logger = logging.getLogger(__name__)

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


def _bbox_area(bbox: Sequence[float]) -> float:
    try:
        return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
            0.0, float(bbox[3]) - float(bbox[1])
        )
    except Exception:
        return 0.0


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    return (
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    )


class SuspiciousActivityDetector(BaseAnalyzer):
    """Flags loitering, running and crouching for tracked persons.

    Geometry-first and cheap: loitering and running need only the tracking
    bboxes (the pipeline injects them into :attr:`current_tracks` on every
    sweep); crouching uses MediaPipe Pose on a per-person crop, limited to
    one pose inference per ``analyze`` call and skipped entirely when
    mediapipe is not installed.

    * **loitering** - same ``track_id`` inside the virtual fence zone for
      more than ``loiter_seconds`` (default 30 s); once per in-zone episode.
    * **running** - bbox-center speed (px/s) exceeds ``run_speed_threshold``
      (default 300); re-arms after ``alert_cooldown`` seconds.
    * **crouching** - pose hips below knee keypoints.

    ``analyze`` returns ``{"capability", "events", "count"}``; the pipeline
    turns each event into an alert with ``module="suspicious_activity"``,
    ``severity="warning"``.
    """

    name = "suspicious_activity"

    def __init__(
        self,
        loiter_seconds: float = 30.0,
        run_speed_threshold: float = 300.0,
        alert_cooldown: float = 10.0,
        crouch_enabled: bool = True,
        track_timeout: float = 5.0,
        pose_interval: float = 2.0,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device)
        self.loiter_seconds = max(0.0, float(loiter_seconds))
        self.run_speed_threshold = max(0.0, float(run_speed_threshold))
        self.alert_cooldown = max(0.0, float(alert_cooldown))
        self.crouch_enabled = bool(crouch_enabled)
        self.track_timeout = float(track_timeout)
        self.pose_interval = float(pose_interval)
        #: set by the pipeline (set_zone) once the virtual fence is ready
        self.zone_monitor: Any = None
        #: latest tracking payload tracks; injected by the pipeline worker
        self.current_tracks: List[Dict[str, Any]] = []
        #: injectable clock (monotonic) for deterministic tests
        self._clock: Callable[[], float] = time.monotonic
        self._pose_unavailable = False
        self._pose_warned = False
        # track_id -> motion / alert state
        self._motion: Dict[Any, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ setup
    def set_zone(self, zone_monitor: Any) -> None:
        """Attach the virtual-fence monitor used for loitering detection."""
        self.zone_monitor = zone_monitor

    # ------------------------------------------------------------------- pose
    def _ensure_pose(self) -> Optional[Any]:
        """Lazily build the MediaPipe Pose model; None when unavailable."""
        if self._model is not None:
            return self._model
        try:
            import mediapipe as mp  # heavy, lazy
        except Exception:
            self._pose_unavailable = True
            if not self._pose_warned:
                self._pose_warned = True
                logger.warning(
                    "mediapipe not installed - crouching detection disabled "
                    "(loitering + running remain active)"
                )
            return None
        pose_module = mp.solutions.pose
        self._model = pose_module.Pose(
            static_image_mode=True,  # per-crop, single image
            model_complexity=1,
            min_detection_confidence=0.5,
        )
        return self._model

    def _crouch_check(self, frame: np.ndarray, bbox: Sequence[float]) -> Optional[bool]:
        """True when hips sit below knees in the person crop, None if unknown."""
        pose = self._ensure_pose()
        if pose is None:
            return None
        height, width = frame.shape[:2]
        x1 = max(0, int(float(bbox[0]) - (float(bbox[2]) - float(bbox[0])) * 0.1))
        y1 = max(0, int(float(bbox[1]) - (float(bbox[3]) - float(bbox[1])) * 0.1))
        x2 = min(width, int(float(bbox[2]) + (float(bbox[2]) - float(bbox[0])) * 0.1))
        y2 = min(height, int(float(bbox[3]) + (float(bbox[3]) - float(bbox[1])) * 0.1))
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None
        crop = frame[y1:y2, x1:x2]
        result = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks is None:
            return None
        landmarks = result.pose_landmarks.landmark
        hip_ys = [
            landmarks[idx].y
            for idx in (_KEYPOINTS["left_hip"], _KEYPOINTS["right_hip"])
            if idx < len(landmarks) and landmarks[idx].visibility >= 0.5
        ]
        knee_ys = [
            landmarks[idx].y
            for idx in (_KEYPOINTS["left_knee"], _KEYPOINTS["right_knee"])
            if idx < len(landmarks) and landmarks[idx].visibility >= 0.5
        ]
        if not hip_ys or not knee_ys:
            return None
        # y grows downward: hips below knees => crouching
        return (sum(hip_ys) / len(hip_ys)) > (sum(knee_ys) / len(knee_ys))

    # --------------------------------------------------------------- analysis
    def _load_model(self) -> Any:  # MediaPipe loads lazily in _crouch_check
        return None

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        now = self._clock()
        events: List[Dict[str, Any]] = []
        # Biggest bbox first: it wins the (single) pose-inference slot.
        tracks = sorted(
            self.current_tracks,
            key=lambda t: _bbox_area(t.get("bbox") or ()),
            reverse=True,
        )
        pose_budget = 1
        for track in tracks:
            track_id = track.get("track_id")
            bbox = track.get("bbox")
            if track_id is None or not bbox or len(bbox) < 4:
                continue
            state = self._motion.setdefault(
                track_id,
                {
                    "center": None,
                    "t": None,
                    "inside_since": None,
                    "loiter_fired": False,
                    "last_run_alert": float("-inf"),
                    "last_crouch_alert": float("-inf"),
                    "last_pose_check": float("-inf"),
                },
            )
            center = _bbox_center(bbox)
            dt = None if state["t"] is None else now - state["t"]
            speed = (
                None
                if dt is None or dt <= 0
                else (
                    (center[0] - state["center"][0]) ** 2
                    + (center[1] - state["center"][1]) ** 2
                )
                ** 0.5
                / dt
            )
            state["center"], state["t"] = center, now

            # -- loitering (needs the virtual fence) -----------------------
            zone_name = getattr(self.zone_monitor, "zone_name", "zone")
            if self.zone_monitor is not None and self.zone_monitor.is_inside(bbox):
                if state["inside_since"] is None:
                    state["inside_since"] = now
                duration = now - state["inside_since"]
                if duration >= self.loiter_seconds and not state["loiter_fired"]:
                    state["loiter_fired"] = True  # once per in-zone episode
                    events.append(
                        {
                            "type": "loitering",
                            "track_id": track_id,
                            "bbox": list(bbox),
                            "duration_s": round(duration, 1),
                            "message": (
                                f"Loitering: person {track_id} inside zone "
                                f"'{zone_name}' for {duration:.0f}s"
                            ),
                        }
                    )
            else:
                # left the zone (or no zone): reset the loiter episode
                state["inside_since"] = None
                state["loiter_fired"] = False

            # -- running ---------------------------------------------------
            if (
                speed is not None
                and speed >= self.run_speed_threshold
                and now - state["last_run_alert"] >= self.alert_cooldown
            ):
                state["last_run_alert"] = now
                events.append(
                    {
                        "type": "running",
                        "track_id": track_id,
                        "bbox": list(bbox),
                        "speed_px_s": round(speed, 1),
                        "message": (
                            f"Running: person {track_id} at {speed:.0f} px/s "
                            f"(threshold {self.run_speed_threshold:.0f})"
                        ),
                    }
                )

            # -- crouching (rate-limited MediaPipe pose) -------------------
            if (
                self.crouch_enabled
                and not self._pose_unavailable
                and pose_budget > 0
                and now - state["last_pose_check"] >= self.pose_interval
                and now - state["last_crouch_alert"] >= self.alert_cooldown
            ):
                state["last_pose_check"] = now
                crouched = self._crouch_check(frame, bbox)
                if crouched is not None:
                    pose_budget -= 1
                    if crouched:
                        state["last_crouch_alert"] = now
                        events.append(
                            {
                                "type": "crouching",
                                "track_id": track_id,
                                "bbox": list(bbox),
                                "message": (
                                    f"Crouching: person {track_id} (hips below knees)"
                                ),
                            }
                        )

        # Drop state for tracks that vanished (person left / ID switched).
        alive = {t.get("track_id") for t in tracks}
        self._motion = {tid: st for tid, st in self._motion.items() if tid in alive}
        return {"capability": self.name, "events": events, "count": len(events)}

    def close(self) -> None:
        if self._model is not None:
            try:
                self._model.close()
            except Exception:
                pass
        super().close()
