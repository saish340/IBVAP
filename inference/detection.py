"""Object detection + multi-object tracking engine (YOLOv8n + ByteTrack).

``DetectionEngine`` is the full-featured detection pipeline used by the
analytics layer:

1. a pretrained Ultralytics YOLOv8n model detects objects on a BGR frame;
2. detections are filtered to surveillance/traffic classes
   (person, car, motorcycle, bus, truck);
3. ByteTrack (via the ``supervision`` library) assigns persistent track IDs
   across frames;
4. ``process_frame`` returns plain dicts (class, confidence, bbox,
   track_id) and can draw boxes + IDs onto the frame for visualization.

Run as a script for a live demo on the fake CCTV stream (Prompt 2, see
scripts/fake_cctv.py):

    python inference/detection.py                       # rtsp://127.0.0.1:8554/cctv
    python inference/detection.py --no-window --frames 200
    python inference/detection.py rtsp://user:pass@camera:554/stream1

Requires: ultralytics, supervision (>= 0.20 recommended), opencv-python.
Model weights (yolov8n.pt, ~6 MB) download automatically on first use.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# COCO classes tracked by this engine (surveillance / traffic focus).
TRACK_CLASSES: Dict[str, int] = {
    "person": 0,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}
TRACK_CLASS_IDS = np.array(sorted(TRACK_CLASSES.values()))
_CLASS_ID_TO_NAME = {v: k for k, v in TRACK_CLASSES.items()}

#: default source for ``python inference/detection.py`` - the fake CCTV
#: stream started by scripts/fake_cctv.py (Prompt 2).
DEFAULT_RTSP_URL = "rtsp://127.0.0.1:8554/cctv"


class DetectionEngine:
    """YOLOv8 detection + ByteTrack tracking on single video frames.

    Example:
        engine = DetectionEngine()
        detections = engine.process_frame(frame)
        annotated = engine.annotated_frame   # frame with boxes + track IDs
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        device: Optional[str] = None,
        draw: bool = True,
    ) -> None:
        # Heavy dependencies are imported lazily so importing this module
        # stays cheap (consistent with the other inference modules).
        import supervision as sv
        from ultralytics import YOLO

        self._sv = sv
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.draw_enabled = draw

        #: persistent ByteTrack tracker (one instance per video/stream!)
        # supervision >= 0.28 deprecates ByteTrack (removal planned in 0.31,
        # no replacement shipped yet) - silence just that warning and pin
        # ``supervision<0.31`` in requirements.txt until an API lands.
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*ByteTrack.*deprecated.*")
            self.byte_tracker = sv.ByteTrack()

        # supervision >= 0.20 splits box drawing and label drawing into two
        # annotators; older versions draw labels via BoxAnnotator directly.
        self._box_annotator = sv.BoxAnnotator(thickness=2)
        self._label_annotator = (
            sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
            if hasattr(sv, "LabelAnnotator")
            else None
        )

        #: last annotated frame produced by :meth:`process_frame` (or None
        #: when drawing was disabled for that call).
        self.annotated_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ API
    def process_frame(
        self, frame: np.ndarray, draw: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Run detect -> class filter -> ByteTrack on one BGR frame.

        Args:
            frame: BGR image (e.g. from ``cv2.VideoCapture.read``).
            draw: draw boxes + track IDs onto the frame (defaults to the
                value chosen at construction time).

        Returns:
            List of dicts, one per tracked object::

                {"class": "person", "confidence": 0.87,
                 "bbox": [x1, y1, x2, y2], "track_id": 3}

            ``track_id`` is ``None`` for detections ByteTrack has not yet
            confirmed (provisional on their first frame).  The annotated
            frame is available as ``self.annotated_frame``.
        """
        if draw is None:
            draw = self.draw_enabled
        detections = self._detect_and_track(frame)
        self.annotated_frame = self._annotate(frame, detections) if draw else None
        return self._to_records(detections)

    # ------------------------------------------------------------- internals
    def _detect_and_track(self, frame: np.ndarray):
        """YOLOv8 predict -> supervision.Detections -> class filter -> ByteTrack."""
        sv = self._sv
        result = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=list(TRACK_CLASS_IDS),  # filter at the detector level
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        # Explicit filter to the tracked classes (no-op when the detector
        # already filtered via ``classes=`` above, but keeps the contract
        # clear and guards against models with different class tables).
        detections = detections[np.isin(detections.class_id, TRACK_CLASS_IDS)]
        # ByteTrack assigns persistent IDs across successive calls.
        return self.byte_tracker.update_with_detections(detections)

    def _to_records(self, detections) -> List[Dict[str, Any]]:
        """Convert a supervision.Detections batch into plain dicts."""
        records: List[Dict[str, Any]] = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = (round(float(v), 1) for v in detections.xyxy[i])
            class_id = int(detections.class_id[i])
            confidence = (
                round(float(detections.confidence[i]), 4)
                if detections.confidence is not None
                else None
            )
            track_id = None
            if detections.tracker_id is not None:
                tid = int(detections.tracker_id[i])
                track_id = tid if tid >= 0 else None  # -1 = not yet confirmed
            records.append(
                {
                    "class": _CLASS_ID_TO_NAME.get(
                        class_id, str(self.model.names.get(class_id, class_id))
                    ),
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "track_id": track_id,
                }
            )
        return records

    def _annotate(self, frame: np.ndarray, detections) -> np.ndarray:
        """Draw bounding boxes + '#id class conf' labels on the frame."""
        labels: List[str] = []
        for i in range(len(detections)):
            tid = -1
            if detections.tracker_id is not None:
                tid = int(detections.tracker_id[i])
            class_id = int(detections.class_id[i])
            name = _CLASS_ID_TO_NAME.get(class_id, str(class_id))
            conf = (
                float(detections.confidence[i])
                if detections.confidence is not None
                else 0.0
            )
            labels.append(f"#{tid} {name} {conf:.2f}")

        # supervision annotators draw in place - work on a copy so the
        # caller's frame is never mutated as a side effect.
        scene = frame.copy()
        if self._label_annotator is not None:  # supervision >= 0.20
            scene = self._box_annotator.annotate(scene=scene, detections=detections)
            scene = self._label_annotator.annotate(
                scene=scene, detections=detections, labels=labels
            )
        else:  # supervision < 0.20: BoxAnnotator draws labels itself
            scene = self._box_annotator.annotate(
                scene=scene, detections=detections, labels=labels
            )
        return scene

# ----------------------------------------------------- live demo (__main__)
def _open_capture(url: str, attempts: int = 10, delay: float = 1.0):
    """Open an RTSP stream / video file, retrying while the server starts."""
    # Ask OpenCV's bundled FFmpeg for a reliable RTSP transport.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = None
    for attempt in range(1, attempts + 1):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                return cap
        cap.release()
        print(f"[detection] connecting to {url} ({attempt}/{attempts}) ...", flush=True)
        time.sleep(delay)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live YOLOv8n + ByteTrack detection on an RTSP stream.",
    )
    parser.add_argument(
        "url", nargs="?", default=DEFAULT_RTSP_URL,
        help="RTSP url or video file path (default: the fake CCTV stream)",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics weights")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frames (0 = run until 'q'/'Esc')")
    parser.add_argument("--no-window", action="store_true",
                        help="headless: print stats only, no GUI")
    parser.add_argument("--connect-attempts", type=int, default=10)
    args = parser.parse_args()

    cap = _open_capture(args.url, attempts=args.connect_attempts)
    if cap is None:
        print(
            f"error: could not open {args.url}\n"
            "If this is the fake CCTV stream, start it first:\n"
            "  python scripts/fake_cctv.py start samples/demo.mp4",
            file=sys.stderr,
        )
        return 1

    engine = DetectionEngine(model_path=args.model, confidence=args.confidence)
    print(
        f"[detection] model '{args.model}' loaded - "
        f"tracking classes: {', '.join(TRACK_CLASSES)}"
    )
    print(
        f"[detection] source: {args.url}"
        + ("" if args.no_window else "  -  press 'q' in the window to quit")
    )

    frames = 0
    fps_frames = 0
    fps = 0.0
    fps_last = time.perf_counter()
    failures = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 50:
                    print("error: stream stopped delivering frames", file=sys.stderr)
                    return 1
                time.sleep(0.05)
                continue
            failures = 0

            started = time.perf_counter()
            detections = engine.process_frame(frame)
            inference_ms = (time.perf_counter() - started) * 1000
            frames += 1
            fps_frames += 1
            now = time.perf_counter()
            if now - fps_last >= 1.0:
                fps = fps_frames / (now - fps_last)
                fps_frames = 0
                fps_last = now

            if args.no_window:
                if frames == 1 or frames % 25 == 0:
                    summary = (
                        ", ".join(
                            f"{d['class']}#{d['track_id']}" for d in detections
                        )
                        or "no tracked objects"
                    )
                    print(
                        f"[detection] frame {frames:5d}  {fps:5.1f} fps  "
                        f"{inference_ms:6.1f} ms  | {summary}"
                    )
            else:
                annotated = (
                    engine.annotated_frame
                    if engine.annotated_frame is not None
                    else frame
                )
                h, w = annotated.shape[:2]
                cv2.putText(
                    annotated,
                    f"{fps:5.1f} fps   {inference_ms:5.0f} ms   "
                    f"{len(detections)} objects",
                    (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
                )
                cv2.imshow("IBVAP detection (YOLOv8n + ByteTrack)", annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            if args.frames and frames >= args.frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[detection] processed {frames} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())


