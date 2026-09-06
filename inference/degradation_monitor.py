"""Lightweight video-quality degradation monitoring.

The monitor uses inexpensive image statistics so it can run before the main
inference stack without competing with the AI models. It does not mutate the
input frame.

Example::

    monitor = ConditionMonitor()
    report = monitor.process_frame(frame)
    print(report.as_dict())

Demo::

    python inference/degradation_monitor.py --source samples/demo.mp4 --overlay
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Optional, Union

import cv2
import numpy as np

# Thresholds are intentionally plain constants so they are easy to tune for a
# camera, then can be overridden per ConditionMonitor instance.
DEFAULT_BLUR_THRESHOLD = 80.0
# Once blur has been confirmed, require a materially sharper image before
# clearing it.  This hysteresis prevents boundary values (for example 79/82)
# from causing condition churn.
DEFAULT_BLUR_CLEAR_THRESHOLD = 90.0
DEFAULT_LOW_LIGHT_THRESHOLD = 50.0
DEFAULT_LOW_CONTRAST_THRESHOLD = 25.0
DEFAULT_NOISE_THRESHOLD = 12.0
DEFAULT_HISTORY_SIZE = 10

CONDITION_CLEAR = "CLEAR"
CONDITION_LOW_LIGHT = "LOW_LIGHT"
CONDITION_BLURRY = "BLURRY"
CONDITION_LOW_CONTRAST = "LOW_CONTRAST/FOGGY"
CONDITION_NOISY = "NOISY"
CONDITION_MULTIPLE = "MULTIPLE_DEGRADED"


@dataclass(frozen=True)
class DegradationReport:
    """Quality classification and measurements for one frame."""

    condition: str
    severity: float
    raw_metrics: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of this report."""
        return asdict(self)


class ConditionMonitor:
    """Classify frame quality and smooth transient classifications.

    Args:
        blur_threshold: variance-of-Laplacian below this is blurry.
        low_light_threshold: mean grayscale intensity below this is low light.
        low_contrast_threshold: grayscale standard deviation below this is
            low contrast/foggy.
        noise_threshold: high-frequency energy above this is noisy.
        history_size: number of recent candidate conditions used for smoothing.
    """

    def __init__(
        self,
        blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
        low_light_threshold: float = DEFAULT_LOW_LIGHT_THRESHOLD,
        low_contrast_threshold: float = DEFAULT_LOW_CONTRAST_THRESHOLD,
        noise_threshold: float = DEFAULT_NOISE_THRESHOLD,
        history_size: int = DEFAULT_HISTORY_SIZE,
        switch_frames: int = 3,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        if switch_frames < 1:
            raise ValueError("switch_frames must be at least 1")
        self.blur_threshold = float(blur_threshold)
        self.low_light_threshold = float(low_light_threshold)
        self.low_contrast_threshold = float(low_contrast_threshold)
        self.noise_threshold = float(noise_threshold)
        self.history: Deque[str] = deque(maxlen=int(history_size))
        # The old majority vote used the current frame to break ties.  At a
        # threshold boundary that made the published condition alternate on
        # successive frames.  Keep a stable state and require a short,
        # consecutive run before transitioning to another condition.
        self.switch_frames = int(switch_frames)
        self._stable_condition: Optional[str] = None
        self._pending_condition: Optional[str] = None
        self._pending_count = 0

    # --------------------------------------------------------------- public API
    def process_frame(self, frame: np.ndarray) -> DegradationReport:
        """Compute metrics and return a smoothed degradation report."""
        gray = self._to_gray(frame)
        metrics = self._metrics(gray)
        candidate, severity = self._classify(metrics)
        self.history.append(candidate)
        condition = self._smoothed_condition(candidate)
        return DegradationReport(
            condition=condition,
            severity=round(float(np.clip(severity, 0.0, 1.0)), 4),
            raw_metrics={key: round(float(value), 4) for key, value in metrics.items()},
        )

    def overlay(
        self,
        frame: np.ndarray,
        report: Optional[DegradationReport] = None,
    ) -> np.ndarray:
        """Return a copy of ``frame`` with condition and severity text."""
        if report is None:
            report = self.process_frame(frame)
        output = frame.copy()
        color = (0, 200, 0) if report.condition == CONDITION_CLEAR else (0, 165, 255)
        if report.condition == CONDITION_MULTIPLE:
            color = (0, 0, 255)
        text = f"{report.condition} | severity {report.severity:.2f}"
        cv2.rectangle(output, (8, 8), (min(output.shape[1] - 8, 440), 42), (0, 0, 0), -1)
        cv2.putText(
            output,
            text,
            (16, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
        return output

    def reset(self) -> None:
        """Clear the rolling classification history."""
        self.history.clear()
        self._stable_condition = None
        self._pending_condition = None
        self._pending_count = 0

    # ------------------------------------------------------------- measurements
    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raise ValueError("frame must be a grayscale or BGR image")

    @staticmethod
    def _metrics(gray: np.ndarray) -> Dict[str, float]:
        """Compute the four requested quality signals."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        high_frequency = gray.astype(np.float32) - blurred.astype(np.float32)
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        histogram /= max(float(histogram.sum()), 1.0)
        return {
            "blur_score": float(laplacian.var()),
            "brightness": float(gray.mean()),
            "contrast": float(gray.std()),
            "noise_energy": float(np.mean(np.abs(high_frequency))),
            "dark_pixel_ratio": float(histogram[:26].sum()),
        }

    def _classify(self, metrics: Dict[str, float]) -> tuple[str, float]:
        degraded = []
        if metrics["brightness"] < self.low_light_threshold:
            degraded.append(CONDITION_LOW_LIGHT)
        blur_limit = (
            DEFAULT_BLUR_CLEAR_THRESHOLD
            if self._stable_condition == CONDITION_BLURRY
            else self.blur_threshold
        )
        if metrics["blur_score"] < blur_limit:
            degraded.append(CONDITION_BLURRY)
        if metrics["contrast"] < self.low_contrast_threshold:
            degraded.append(CONDITION_LOW_CONTRAST)
        if metrics["noise_energy"] > self.noise_threshold:
            degraded.append(CONDITION_NOISY)

        if not degraded:
            return CONDITION_CLEAR, 0.0

        severity_values = [self._severity_for(condition, metrics) for condition in degraded]
        severity = max(severity_values)
        if len(degraded) > 1:
            return CONDITION_MULTIPLE, severity
        return degraded[0], severity

    def _severity_for(self, condition: str, metrics: Dict[str, float]) -> float:
        if condition == CONDITION_LOW_LIGHT:
            return 1.0 - metrics["brightness"] / max(self.low_light_threshold, 1.0)
        if condition == CONDITION_BLURRY:
            return 1.0 - metrics["blur_score"] / max(self.blur_threshold, 1.0)
        if condition == CONDITION_LOW_CONTRAST:
            return 1.0 - metrics["contrast"] / max(self.low_contrast_threshold, 1.0)
        if condition == CONDITION_NOISY:
            return metrics["noise_energy"] / max(self.noise_threshold, 1.0) - 1.0
        return 0.0

    def _smoothed_condition(self, current: str) -> str:
        """Publish a stable condition after consecutive supporting samples."""
        if self._stable_condition is None:
            self._stable_condition = current
            return current
        if current == self._stable_condition:
            self._pending_condition = None
            self._pending_count = 0
            return self._stable_condition
        if current == self._pending_condition:
            self._pending_count += 1
        else:
            self._pending_condition = current
            self._pending_count = 1
        if self._pending_count >= self.switch_frames:
            self._stable_condition = current
            self._pending_condition = None
            self._pending_count = 0
        return self._stable_condition


def _open_source(source: str) -> Union[int, str]:
    return int(source) if source.isdigit() else source


def main() -> int:
    parser = argparse.ArgumentParser(description="Live video degradation monitor")
    parser.add_argument("--source", default="0", help="webcam index, video path, or RTSP URL")
    parser.add_argument("--history-size", type=int, default=DEFAULT_HISTORY_SIZE)
    parser.add_argument("--overlay", action="store_true", help="draw condition and severity on the frame")
    parser.add_argument("--no-window", action="store_true", help="print reports without opening a window")
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames; 0 runs until q")
    args = parser.parse_args()

    capture = cv2.VideoCapture(_open_source(args.source))
    if not capture.isOpened():
        print(f"error: could not open {args.source}")
        return 1

    monitor = ConditionMonitor(history_size=args.history_size)
    frame_count = 0
    last_report = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            report = monitor.process_frame(frame)
            frame_count += 1
            if report.condition != last_report:
                print(
                    f"[DEGRADATION] frame={frame_count} condition={report.condition} "
                    f"severity={report.severity:.2f} metrics={report.raw_metrics}",
                    flush=True,
                )
                last_report = report.condition

            if not args.no_window:
                display = monitor.overlay(frame, report) if args.overlay else frame
                cv2.imshow("IBVAP degradation monitor", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.frames and frame_count >= args.frames:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
