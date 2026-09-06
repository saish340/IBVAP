"""End-to-end RTSP analytics pipeline.

Ties together the whole IBVAP stack on one continuous loop:

1. connect to an RTSP stream (retrying while the server / feed comes up);
2. for every **3rd** frame (``PROCESS_EVERY_N=3``) run, in order:

   - :class:`inference.detection.DetectionEngine` (Prompt 3)
        -> tracked detections ``{class, confidence, bbox, track_id}``
   - :class:`inference.virtual_fence.ZoneMonitor` (Prompt 5)
        -> one-shot intrusion events ``{track_id, class, timestamp, zone_name}``
   - :class:`inference.face_verification.FaceVerificationEngine` (Prompt 4)
        -> face matches ``{bbox, name, confidence, face_confidence}``

3. any intrusion event **or** a face matched against a **flagged** watchlist
   entry is POSTed to the backend's ``/events/ingest`` as an alert, with a
   JPEG thumbnail base64-encoded, so it arrives live on ``/ws/alerts``;
4. the loop is resilient: a dropped/None frame is skipped, a single engine or
   POST failure is logged and does not crash the pipeline, and the camera
   connection is retried automatically.

Run it:

    python -m ingestion.pipeline --source rtsp://127.0.0.1:8554/cctv
    python -m ingestion.pipeline --source 0 --no-faces
    python -m ingestion.pipeline --source 0 --polygon 100,300 700,300 700,700 100,700

Note: the face engine needs deepface/TensorFlow, which only installs on
Python <= 3.13. Pass ``--no-faces`` on 3.14 or when keeping the pipeline lean.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests  # part of the FastAPI/uvicorn stack

from inference.degradation_monitor import (
    CONDITION_BLURRY,
    CONDITION_CLEAR,
    CONDITION_LOW_LIGHT,
    CONDITION_NOISY,
    ConditionMonitor,
    DegradationReport,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

#: how often (every Nth frame) the inference engines run on the loop
PROCESS_EVERY_N = 3
#: severity for intrusion (virtual fence) alerts
FENCE_SEVERITY = "critical"
#: severity for a flagged-watchlist face match
FACE_SEVERITY = "warning"

#: engine module name sent to /events/ingest + shown in alert telemetry
MODULE_DETECTION = "detection"
MODULE_FENCE = "fence"
MODULE_FACE = "face_verification"

#: names considered "flagged" (sanity-guard). Real flag lists live in the
#: backend watchlist API; here we treat every enrolled name as flagged unless
#: ``--flag-names`` restricts it.
FLAGGED_NAME = "flagged"  # sentinel in the watchlist DB, if present


# ------------------------------------------------------------------ config
@dataclass
class PipelineConfig:
    """Runtime configuration for the end-to-end pipeline."""

    source: str
    api_url: str = "http://localhost:8000"
    camera_id: str = "cam-01"
    process_every_n: int = PROCESS_EVERY_N

    # engines
    yolo_model: str = "yolov8n.pt"
    confidence: float = 0.35
    enable_faces: bool = True
    detector_backend: str = "retinaface"
    face_model: str = "ArcFace"
    face_threshold: float = 0.6
    watchlist_db: str = "watchlist.db"
    flagged_names: Sequence[str] = ()
    # if set, ignores the flag list and alerts on every watchlist match
    alert_all_faces: bool = False
    night_enhance: bool = True
    brightness_threshold: int = 50
    zero_dce_weights: Optional[str] = None
    pending_events_db: str = "pending_events.db"
    degradation_history_size: int = 10
    degraded_consensus_frames: int = 3
    degraded_confidence_scale: float = 0.65

    # zone
    polygon: Sequence[Tuple[float, float]] = ()
    zone_name: str = "restricted"
    inside_threshold: int = 2

    # resilience
    reconnect_delay: float = 2.0
    max_frames: int = 0  # 0 = run forever
    max_alerts: Optional[int] = None  # stop after this many alerts (tests)


# ------------------------------------------------------------ alert client
class AlertClient:
    """HTTP alert client with durable SQLite buffering while offline."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        pending_db: str = "pending_events.db",
        health_interval: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self.pending_db = pending_db
        self._offline = False
        self._stop_event = threading.Event()
        self._init_pending_db()
        self._worker = threading.Thread(
            target=self._connectivity_loop,
            args=(health_interval,),
            name="ibvap-alert-buffer",
            daemon=True,
        )
        self._worker.start()

    @property
    def ingest_url(self) -> str:
        return f"{self.base_url}/events/ingest"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    def close(self) -> None:
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._session.close()

    def _init_pending_db(self) -> None:
        with sqlite3.connect(self.pending_db) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS pending_events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL, "
                "created_at REAL NOT NULL)"
            )

    def _queue(self, payload: Dict[str, Any]) -> None:
        with sqlite3.connect(self.pending_db) as db:
            db.execute(
                "INSERT INTO pending_events (payload, created_at) VALUES (?, ?)",
                (json.dumps(payload), time.time()),
            )

    def _pending(self) -> List[Tuple[int, str]]:
        with sqlite3.connect(self.pending_db) as db:
            return db.execute(
                "SELECT id, payload FROM pending_events ORDER BY id"
            ).fetchall()

    def _delete_pending(self, event_id: int) -> None:
        with sqlite3.connect(self.pending_db) as db:
            db.execute("DELETE FROM pending_events WHERE id = ?", (event_id,))

    def _send(self, payload: Dict[str, Any]) -> bool:
        try:
            response = self._session.post(self.ingest_url, json=payload, timeout=10)
        except requests.RequestException:
            return False
        return 200 <= response.status_code < 300

    def _flush(self) -> bool:
        for event_id, serialized in self._pending():
            try:
                payload = json.loads(serialized)
            except json.JSONDecodeError:
                logger.error("discarding malformed pending event %s", event_id)
                self._delete_pending(event_id)
                continue
            if not self._send(payload):
                return False
            self._delete_pending(event_id)
        return True

    def _connectivity_loop(self, interval: float) -> None:
        while not self._stop_event.wait(interval):
            try:
                response = self._session.get(self.health_url, timeout=3)
                online = 200 <= response.status_code < 300
            except requests.RequestException:
                online = False
            if online:
                if self._offline:
                    logger.info("backend online - flushing pending events")
                flushed = self._flush()
                if self._offline and flushed:
                    logger.info("online mode restored")
                self._offline = not flushed
            elif not self._offline:
                self._offline = True
                logger.warning("backend offline - buffering events locally")

    def post_alert(
        self,
        *,
        module: str,
        severity: str,
        message: str,
        track_id: Optional[int],
        timestamp: float,
        camera_id: str,
        thumbnail_base64: Optional[str] = None,
        detection_condition: str = CONDITION_CLEAR,
        detection_reliability_score: float = 1.0,
    ) -> bool:
        """POST one alert; returns True on success (2xx). Never raises."""
        payload = {
            "module": module,
            "severity": severity,
            "message": message,
            "track_id": track_id,
            "timestamp": timestamp,
            "camera_id": camera_id,
            "thumbnail_base64": thumbnail_base64,
            "detection_condition": detection_condition,
            "detection_reliability_score": round(float(detection_reliability_score), 4),
        }
        if not self._send(payload):
            self._queue(payload)
            if not self._offline:
                self._offline = True
                logger.warning("backend offline - buffering events locally")
            return False
        return True

# ------------------------------------------------------------- thumbnail
def encode_jpeg_thumbnail(
    frame: np.ndarray, max_width: int = 640, quality: int = 60
) -> Optional[str]:
    """Resize ``frame`` and return a base64 JPEG string (or None on error)."""
    if frame is None:
        return None
    try:
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        logger.exception("thumbnail encoding failed")
        return None


# -------------------------------------------------------------- zone helper
def default_polygon(w: int, h: int) -> List[Tuple[float, float]]:
    """A sensible default restricted zone: lower-center third of the frame."""
    return [(0.1 * w, 0.35 * h), (0.9 * w, 0.35 * h), (0.9 * w, 0.95 * h), (0.1 * w, 0.95 * h)]


def parse_polygon(pairs: Sequence[str]) -> List[Tuple[float, float]]:
    """Parse ``["x,y", "x,y", ...]`` into float vertices."""
    pts = []
    for token in pairs:
        parts = token.replace("(", "").replace(")", "").split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"bad polygon vertex {token!r} (want 'x,y')")
        pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 3:
        raise argparse.ArgumentTypeError("polygon needs at least 3 vertices")
    return pts


# ------------------------------------------------------------ engine factory
def build_engines(
    cfg: PipelineConfig, frame_size: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """Build all three engines for this pipeline's configuration.

    Returns a dict with keys ``detection``, ``zone`` and (if enabled)
    ``face``. Face engine construction fails loudly (missing deepface) so
    the caller can fall back to ``--no-faces``.
    """
    from inference.detection import DetectionEngine
    from inference.virtual_fence import ZoneMonitor

    detection = DetectionEngine(model_path=cfg.yolo_model, confidence=cfg.confidence)

    if cfg.polygon:
        polygon: Sequence[Tuple[float, float]] = cfg.polygon
    elif frame_size is not None:
        w, h = frame_size
        polygon = default_polygon(w, h)
    else:
        polygon = default_polygon(1280, 720)

    zone = ZoneMonitor(
        polygon,
        zone_name=cfg.zone_name,
        inside_threshold=cfg.inside_threshold,
    )

    engines: Dict[str, Any] = {"detection": detection, "zone": zone}
    if cfg.enable_faces:
        from inference.face_verification import FaceVerificationEngine

        engines["face"] = FaceVerificationEngine(
            db_path=cfg.watchlist_db,
            threshold=cfg.face_threshold,
            detector_backend=cfg.detector_backend,
            model_name=cfg.face_model,
            draw=False,
        )
    return engines


def load_flagged_names(cfg: PipelineConfig) -> set:
    """Names considered flagged (alert on match). ``alert_all_faces`` -> all."""
    if cfg.alert_all_faces:
        return {"*"}
    if cfg.flagged_names:
        return set(cfg.flagged_names)
    if cfg.watchlist_db or FLAGGED_NAME:  # sentinel or default — treat all names flagged
        return {FLAGGED_NAME}  # marker: match against the DB's enrolled names instead
    return set()

# --------------------------------------------------------------- pipeline
class RTSPPipeline:
    """Continuous RTSP analytics loop with resilience guards."""

    def __init__(
        self,
        cfg: PipelineConfig,
        engines: Optional[Dict[str, Any]] = None,
        alert_client: Optional[AlertClient] = None,
    ) -> None:
        self.cfg = cfg
        self.alert_client = alert_client or AlertClient(cfg.api_url)
        self.engines = engines or {}
        self.frame_count = 0
        self.processed_count = 0
        self.alert_count = 0
        self._stop = False

        self._flagged = load_flagged_names(cfg)
        # names matched against the DB (used when alerting on all enrolled)
        self._enrolled_names: set = set()
        self.condition_monitor = ConditionMonitor(
            history_size=max(1, cfg.degradation_history_size)
        )
        self.current_degradation = DegradationReport(
            condition=CONDITION_CLEAR,
            severity=0.0,
            raw_metrics={},
        )
        self._consensus_counts: Dict[str, int] = {}
        self._consensus_last_frame: Dict[str, int] = {}
        self._pending_fence_events: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------ lifecycle
    def stop(self) -> None:
        """Request a clean stop (checked between frames)."""
        self._stop = True

    # ------------------------------------------------------------ equipment
    def _open_capture(self, source: str, attempts: int = 10, delay: Optional[float] = None):
        """Open ``source`` (RTSP/webcam/file), retrying while it comes up."""
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        delay = delay if delay is not None else self.cfg.reconnect_delay
        cap = None
        for attempt in range(1, attempts + 1):
            try:
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok:
                        return cap
            except Exception as exc:
                logger.warning("capture open error: %s", exc)
            if cap is not None:
                cap.release()
                cap = None
            logger.warning("source not ready (%s), retrying in %.1fs ...", source, delay)
            time.sleep(delay)
        return None

    def _ensure_engines(self) -> None:
        """Build any engines we do not have yet (after cap exposes frame size)."""
        if not self.engines:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            self.engines = build_engines(self.cfg, frame_size=(w, h))

    # -------------------------------------------------------------- run loop
    def run(self) -> int:
        """Run the continuous loop until stop() / max_frames / max_alerts."""
        cap = self._open_capture(self.cfg.source)
        if cap is None:
            logger.error(
                "could not open source %r after retries - backend reachable?",
                self.cfg.source,
            )
            return 1
        self.cap = cap
        try:
            self._ensure_engines()
            logger.info(
                "pipeline ready on %r (engines: %s)",
                self.cfg.source, ", ".join(sorted(self.engines)),
            )
            while not self._stop:
                ok, frame = cap.read()
                if not ok or frame is None:
                    # dropped frame / transient failure - skip, do not crash
                    logger.warning("frame read failed (skipping)")
                    time.sleep(0.05)
                    continue
                self.frame_count += 1
                degradation = self.condition_monitor.process_frame(frame)

                # every Nth frame -> run the full analysis stack
                if self.frame_count % self.cfg.process_every_n != 0:
                    continue

                self._process_frame(frame, degradation)
                if self.cfg.max_frames and self.processed_count >= self.cfg.max_frames:
                    logger.info("reached max_frames=%s, stopping", self.cfg.max_frames)
                    break
                if self.cfg.max_alerts and self.alert_count >= self.cfg.max_alerts:
                    logger.info("reached max_alerts=%s, stopping", self.cfg.max_alerts)
                    break
        except KeyboardInterrupt:
            logger.info("interrupted, stopping pipeline")
        finally:
            cap.release()
            self.alert_client.close()
        logger.info("pipeline stopped: %d frames, %d processed, %d alerts",
                    self.frame_count, self.processed_count, self.alert_count)
        return 0

# ------------------------------------------------------------- per-frame
    def _process_frame(
        self, frame: np.ndarray, degradation: Optional[DegradationReport] = None
    ) -> None:
        """Run all engines on one frame and alert on any trigger."""
        self.current_degradation = degradation or self.condition_monitor.process_frame(frame)
        condition = self.current_degradation.condition
        degraded = condition in (CONDITION_BLURRY, CONDITION_NOISY) or (
            self.current_degradation.severity > 0.6
        )
        if self.cfg.night_enhance and condition == CONDITION_LOW_LIGHT:
            try:
                from inference.night_enhance import enhance_frame

                frame, method = enhance_frame(
                    frame,
                    brightness_threshold=self.cfg.brightness_threshold,
                    weights_path=self.cfg.zero_dce_weights,
                )
                if method != "passthrough":
                    logger.debug("low-light preprocessing applied: %s", method)
            except Exception:
                logger.exception("night enhancement failed; using original frame")
        detector = self.engines.get("detection")
        if detector is not None and hasattr(detector, "confidence"):
            detector.confidence = (
                self.cfg.confidence * self.cfg.degraded_confidence_scale
                if degraded
                else self.cfg.confidence
            )
        if degraded:
            logger.info(
                "degraded detection mode: condition=%s severity=%.2f confidence=%.3f consensus=%d",
                condition,
                self.current_degradation.severity,
                getattr(detector, "confidence", self.cfg.confidence),
                self.cfg.degraded_consensus_frames,
            )
        try:
            detections = self.engines["detection"].process_frame(frame)
        except Exception as exc:
            logger.exception("detection failed: %s", exc)
            detections = []
        self.processed_count += 1

        self._handle_zone(detections, frame, consensus_required=degraded)
        if self.engines.get("face") is not None:
            self._handle_faces(frame, consensus_required=degraded)

    def _consensus_ready(self, key: str, required: bool) -> bool:
        """Require consecutive processed frames for degraded conditions."""
        if not required:
            return True
        previous = self._consensus_last_frame.get(key)
        count = self._consensus_counts.get(key, 0) + 1 if previous == self.processed_count - 1 else 1
        self._consensus_counts[key] = count
        self._consensus_last_frame[key] = self.processed_count
        return count >= max(1, self.cfg.degraded_consensus_frames)

    def _alert_context(self) -> Dict[str, Any]:
        return {
            "detection_condition": self.current_degradation.condition,
            "detection_reliability_score": 1.0 - self.current_degradation.severity,
        }

    def _handle_zone(
        self,
        detections: List[Dict[str, Any]],
        frame: np.ndarray,
        consensus_required: bool = False,
    ) -> None:
        """Feed detections to ZoneMonitor and POST any intrusion alerts."""
        try:
            events = self.engines["zone"].update(detections)
        except Exception as exc:
            logger.exception("zone update failed: %s", exc)
            return
        for ev in events:
            self._pending_fence_events[int(ev["track_id"])] = ev

        present_ids = {int(d["track_id"]) for d in detections if d.get("track_id") is not None}
        for track_id in list(self._pending_fence_events):
            if track_id not in present_ids:
                continue
            if not self._consensus_ready(f"fence:{track_id}", consensus_required):
                continue
            ev = self._pending_fence_events.pop(track_id)
            thumbnail = encode_jpeg_thumbnail(frame)
            ok = self.alert_client.post_alert(
                module=MODULE_FENCE,
                severity=FENCE_SEVERITY,
                message=(
                    f"{ev.get('class', 'object')}#{ev.get('track_id')} "
                    f"entered zone '{ev.get('zone_name', self.cfg.zone_name)}'"
                ),
                track_id=ev.get("track_id"),
                timestamp=float(ev.get("timestamp", time.time())),
                camera_id=self.cfg.camera_id,
                thumbnail_base64=thumbnail,
                **self._alert_context(),
            )
            if ok:
                self.alert_count += 1
                logger.info("fence alert posted: %s", ev)

    def _handle_faces(self, frame: np.ndarray, consensus_required: bool = False) -> None:
        """Match faces against the watchlist; alert on flagged identities."""
        try:
            faces = self.engines["face"].process_frame(frame)
        except Exception as exc:
            logger.exception("face verification failed: %s", exc)
            return
        if not faces:
            return

        for face in faces:
            name = face.get("name", "UNKNOWN")
            if name == "UNKNOWN":
                continue
            if not self._is_flagged(name):
                continue
            if not self._consensus_ready(f"face:{name}", consensus_required):
                continue
            thumbnail = encode_jpeg_thumbnail(frame)
            ok = self.alert_client.post_alert(
                module=MODULE_FACE,
                severity=FACE_SEVERITY,
                message=f"flagged face '{name}' matched (conf {face.get('confidence', 0.0):.2f})",
                track_id=None,
                timestamp=time.time(),
                camera_id=self.cfg.camera_id,
                thumbnail_base64=thumbnail,
                **self._alert_context(),
            )
            if ok:
                self.alert_count += 1
                logger.info("face alert posted for '%s'", name)

    def _is_flagged(self, name: str) -> bool:
        """Whether a watchlist name should trigger an alert."""
        if "*" in self._flagged:
            return True
        if FLAGGED_NAME in self._flagged:
            # treat every enrolled name as flagged (default behaviour)
            return True
        return name in self._flagged

# --------------------------------------------------------------- CLI (__main__)
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="End-to-end RTSP analytics pipeline: detection + virtual "
                    "fence + face verification -> /events/ingest alerts.",
    )
    parser.add_argument("--source", default="rtsp://127.0.0.1:8554/cctv",
                        help="RTSP url, video file, or webcam index")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="backend base URL (default: %(default)s)")
    parser.add_argument("--camera-id", default="cam-01")
    parser.add_argument("--process-every", type=int, default=PROCESS_EVERY_N,
                        help="analyze every Nth frame (default: %(default)s)")

    # detection
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)

    # zone
    parser.add_argument("--polygon", nargs="*", default=None,
                        help="zone vertices 'x,y ...' (default: lower-center rect)")
    parser.add_argument("--zone-name", default="restricted")
    parser.add_argument("--inside-threshold", type=int, default=2)

    # faces
    parser.add_argument("--no-faces", action="store_true",
                        help="disable the face engine (no deepface/TF needed)")
    parser.add_argument("--watchlist-db", default="watchlist.db",
                        help="watchlist SQLite file (default: watchlist.db)")
    parser.add_argument("--flag-names", nargs="*", default=None,
                        help="names that trigger alerts (default: all enrolled)")
    parser.add_argument("--alert-all-faces", action="store_true",
                        help="alert on every watchlist match regardless of flags")
    parser.add_argument("--no-night-enhance", action="store_true",
                        help="disable automatic low-light CLAHE preprocessing")
    parser.add_argument("--brightness-threshold", type=int, default=50)
    parser.add_argument("--zero-dce-weights", default=None)
    parser.add_argument("--pending-events-db", default="pending_events.db")

    # limits / resilience
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after this many processed frames (0 = forever)")
    parser.add_argument("--max-alerts", type=int, default=None,
                        help="stop after this many alerts (0/None = forever)")
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = PipelineConfig(
        source=args.source,
        api_url=args.api_url,
        camera_id=args.camera_id,
        process_every_n=max(1, args.process_every),
        yolo_model=args.yolo_model,
        confidence=args.confidence,
        enable_faces=not args.no_faces,
        watchlist_db=args.watchlist_db,
        flagged_names=args.flag_names or (),
        alert_all_faces=args.alert_all_faces,
        night_enhance=not args.no_night_enhance,
        brightness_threshold=args.brightness_threshold,
        zero_dce_weights=args.zero_dce_weights,
        pending_events_db=args.pending_events_db,
        polygon=parse_polygon(args.polygon) if args.polygon else (),
        zone_name=args.zone_name,
        inside_threshold=args.inside_threshold,
        reconnect_delay=args.reconnect_delay,
        max_frames=max(0, args.max_frames),
        max_alerts=args.max_alerts,
    )

    if cfg.enable_faces:
        try:
            engines = build_engines(cfg)
        except ImportError as exc:
            logger.warning("face engine unavailable (%s); continuing without faces "
                           "(pass --no-faces to silence this)", exc)
            cfg.enable_faces = False
            engines = build_engines(cfg)
    else:
        engines = build_engines(cfg)

    pipeline = RTSPPipeline(
        cfg=cfg,
        engines=engines,
        alert_client=AlertClient(cfg.api_url, pending_db=cfg.pending_events_db),
    )
    return pipeline.run()


if __name__ == "__main__":
    sys.exit(main())