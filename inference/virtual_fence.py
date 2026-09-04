"""Virtual fence / restricted-zone intrusion detection.

``ZoneMonitor`` checks tracked object detections (as produced by Prompt 3's
``DetectionEngine`` - bbox + track_id per object) against a polygon zone,
and emits one intrusion event per object per crossing using shapely.

The zone is a virtual fence: an event fires the first time an object's
bottom-center point is detected inside the polygon, and not again while the
object stays inside. It re-arms only when the object leaves (or its track
disappears), so repeated entries naturally produce one event per crossing.

Live demo (run against any video / RTSP stream, e.g. the fake CCTV feed):

    python inference/virtual_fence.py
    python inference/virtual_fence.py --source rtsp://127.0.0.1:8554/cctv --no-window --frames 200
    python inference/virtual_fence.py --polygon 100,300 700,300 700,700 100,700

Requires: shapely (pip install shapely), and for the demo: opencv-python,
ultralytics, supervision.

Events:

    {"track_id": 3, "class": "person", "timestamp": 1730000000.0, "zone_name": "restricted"}
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

#: default screen-relative polygon (percent of frame w/h) for the demo
DEFAULT_POLYGON_REL = [(0.1, 0.35), (0.9, 0.35), (0.9, 0.95), (0.1, 0.95)]


class ZoneMonitor:
    """Monitors a polygon zone and reports one intrusion event per object.

    Args:
        polygon: list of ``(x, y)`` vertices of the restricted zone, in the
            same coordinate space as the detections (normal 2D image space).
        zone_name: label used for the zone in returned events.
        inside_threshold: minimum number of consecutive frames an object must
            be inside the zone before an event fires. Catches flicker and
            boundary noise from the tracker.
    """

    def __init__(
        self,
        polygon: Sequence[Tuple[float, float]],
        zone_name: str = "restricted",
        inside_threshold: int = 2,
    ) -> None:
        if len(polygon) < 3:
            raise ValueError("polygon must have at least 3 vertices")
        # shapely Polygon: buffer(0) cleans self-intersections & degenerate
        # edges; ensure_valid optional. Normalize to a canonical ring.
        self.polygon = Polygon(polygon)
        self.polygon = self.polygon.buffer(0)
        if self.polygon.is_empty:
            raise ValueError("polygon is degenerate or empty")
        if not self.polygon.is_valid:
            # last-ditch: build the convex hull so geometry ops are safe
            self.polygon = self.polygon.convex_hull
        self.zone_name = zone_name
        self.inside_threshold = int(inside_threshold)

        self._state: Dict[int, int] = {}       # track_id -> consecutive-inside count
        self._fired: List[int] = []            # track_ids that already fired

    # ------------------------------------------------------------------ API
    def update(
        self, detections: Sequence[Dict[str, Any]], timestamp: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Feed this frame's tracked detections; return newly fired events.

        Each detection dict must contain ``bbox`` ([x1, y1, x2, y2]) and
        ``track_id`` (an int or None). Objects are treated as a single point:
        the **bottom-center** of the bounding box.

        An event is emitted at most once per track (until the track leaves
        the zone, then it re-arms). ``timestamp`` defaults to
        ``time.time()`` if not supplied.
        """
        ts = timestamp if timestamp is not None else time.time()
        events: List[Dict[str, Any]] = []

        present = set()
        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue  # no valid identity -> cannot track state
            track_id = int(track_id)
            present.add(track_id)

        # 1) re-arm tracks that are no longer in the zone (or disappeared)
        active = self._objects_inside(detections)
        now_inside = set(active)
        for track_id in list(self._fired):
            if track_id not in now_inside:
                self._fired.remove(track_id)   # left zone -> re-arm
        for track_id in list(self._state):
            if track_id not in now_inside:
                del self._state[track_id]      # left zone -> reset its counter

        # 2) accumulate presence for objects inside the zone
        for track_id in active:
            if track_id in self._fired:
                continue  # already fired for this crossing
            count = self._state.get(track_id, 0) + 1
            self._state[track_id] = count
            if count >= self.inside_threshold:
                class_name = self._class_for(detections, track_id)
                events.append(
                    {
                        "track_id": track_id,
                        "class": class_name,
                        "timestamp": ts,
                        "zone_name": self.zone_name,
                    }
                )
                self._fired.append(track_id)

        # 3) drop stale state for tracks that are no longer present
        stale = [tid for tid in self._state if tid not in present]
        for tid in stale:
            del self._state[tid]

        return events

# -------------------------------------------------------------- geometry
    def is_inside(self, bbox: Sequence[float]) -> bool:
        """Whether a bbox's bottom-center point is inside the zone."""
        return self.point_inside(self._bottom_center(bbox))

    def point_inside(self, point: Tuple[float, float]) -> bool:
        """Whether ``(x, y)`` is inside the zone polygon (shapely)."""
        return self.polygon.contains(Point(point))

    # ------------------------------------------------------------- internals
    @staticmethod
    def _bottom_center(bbox: Sequence[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        return ((x1 + x2) / 2.0, y2)

    def _objects_inside(self, detections: Sequence[Dict[str, Any]]):
        inside = []
        for detection in detections:
            track_id = detection.get("track_id")
            bbox = detection.get("bbox")
            if track_id is None or bbox is None or len(bbox) < 4:
                continue
            if self.is_inside(bbox):
                inside.append(int(track_id))
        return inside

    @staticmethod
    def _class_for(detections: Sequence[Dict[str, Any]], track_id: int) -> str:
        for detection in detections:
            if detection.get("track_id") == track_id and detection.get("class"):
                return str(detection["class"])
        return "unknown"

    # --------------------------------------------------------- annotate helpers
    def _annotate(
        self,
        frame: np.ndarray,
        detections: Sequence[Dict[str, Any]] = (),
        highlight_zones: bool = True,
    ) -> np.ndarray:
        """Draw the zone + all detection boxes; red if inside the zone."""
        scene = frame.copy()
        if highlight_zones:
            self.draw_polygon(scene, color=(0, 165, 255), thickness=2)  # orange

        for detection in detections:
            track_id = detection.get("track_id")
            bbox = detection.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (int(v) for v in bbox)
            dangerous = track_id is not None and self.is_inside(bbox)
            color = (0, 0, 220) if dangerous else (0, 200, 0)  # red / green
            cv2.rectangle(scene, (x1, y1), (x2, y2), color, 2)
            label = f"{detection.get('class', '?')}#{track_id}"
            if dangerous:
                label += " IN-ZONE"
            cv2.putText(
                scene, label, (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
            # mark the bottom-center point used for the inside test
            cx, cy = self._bottom_center(bbox)
            cv2.circle(scene, (int(cx), int(cy)), 3, (255, 255, 0), -1)
        return scene

    def draw_polygon(
        self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 165, 255),
        thickness: int = 2,
    ) -> None:
        """Draw the zone polygon onto ``frame`` in place."""
        pts = np.array(self.polygon.exterior.coords, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)

# ----------------------------------------------------- live demo (__main__)
def _default_polygon(w: int, h: int) -> List[Tuple[float, float]]:
    return [(x * w, y * h) for x, y in DEFAULT_POLYGON_REL]


def _open_capture(source: str, attempts: int = 10, delay: float = 1.0):
    """Open an RTSP stream / video file / webcam, retrying while it starts."""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = None
    for attempt in range(1, attempts + 1):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                return cap
        cap.release()
        print(f"[fence] connecting to {source} ({attempt}/{attempts}) ...", flush=True)
        time.sleep(delay)
    return None


def _polygon_from_args(pairs: Sequence[str]) -> List[Tuple[float, float]]:
    """Parse ``100,300 700,300 ...`` into [(x, y), ...] floats."""
    pts = []
    for token in pairs:
        parts = token.replace("(", "").replace(")", "").split(",")
        if len(parts) != 2:
            raise ValueError(f"bad polygon vertex {token!r} (want 'x,y')")
        pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 3:
        raise ValueError("polygon needs at least 3 vertices")
    return pts


def main() -> int:
    # Make the project root importable (scripts run from repo root OR
    # `python inference/virtual_fence.py`):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from inference.detection import DetectionEngine  # real engine from Prompt 3

    parser = argparse.ArgumentParser(
        description="Virtual fence demo: YOLOv8n + ByteTrack detections, "
                    "ZoneMonitor intrusion events, red in-zone highlight.",
    )
    parser.add_argument("--source", default="0",
                        help="webcam index, video file, or RTSP url (default: webcam 0)")
    parser.add_argument("--polygon", nargs="*", default=None,
                        help="zone vertices 'x,y ...' (default: lower-center rectangle)")
    parser.add_argument("--zone-name", default="restricted")
    parser.add_argument("--no-window", action="store_true", help="headless stats")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frames (0 = run until 'q'/'Esc')")
    parser.add_argument("--connect-attempts", type=int, default=10)
    args = parser.parse_args()

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = _open_capture(src, attempts=args.connect_attempts)
    if cap is None:
        print(f"error: could not open source {args.source}", file=sys.stderr)
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    polygon = _polygon_from_args(args.polygon) if args.polygon else _default_polygon(w, h)

    monitor = ZoneMonitor(polygon, zone_name=args.zone_name)
    engine = DetectionEngine()  # YOLOv8n + ByteTrack
    print(f"[fence] zone '{args.zone_name}': {[(int(x), int(y)) for x, y in polygon]}")
    print(f"[fence] source: {args.source}"
          + ("" if args.no_window else "  -  press 'q' in the window to quit"))

    frames = 0
    fps_frames = 0
    fps = 0.0
    fps_last = time.perf_counter()
    all_events: List[Dict[str, Any]] = []
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

            events = monitor.update(detections)  # events fired this frame
            if events:
                for ev in events:
                    ts = time.strftime("%H:%M:%S", time.localtime(ev["timestamp"]))
                    print(f"[fence] INTRUSION {ev['class']}#{ev['track_id']} "
                          f"zone '{ev['zone_name']}' at {ts}")
                all_events.extend(events)

            if args.no_window:
                if frames == 1 or frames % 25 == 0:
                    print(f"[fence] frame {frames:5d}  {fps:5.1f} fps  "
                          f"{inference_ms:6.1f} ms  | {len(events)} new event(s), "
                          f"{len(all_events)} total")
            else:
                annotated = monitor._annotate(
                    frame, detections, highlight_zones=True
                )
                hh, ww = annotated.shape[:2]
                cv2.putText(
                    annotated,
                    f"{fps:4.1f} fps  events: {len(all_events)}",
                    (10, hh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
                )
                cv2.imshow("IBVAP virtual fence (ZoneMonitor)", annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            if args.frames and frames >= args.frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[fence] processed {frames} frames, {len(all_events)} intrusion event(s)")
    for ev in all_events:
        print("  ", ev)
    return 0


if __name__ == "__main__":
    sys.exit(main())