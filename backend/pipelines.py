"""Per-stream processing pipelines: glue between ingestion and inference.

A :class:`StreamPipeline` reads frames from a capture backend on one worker
thread, runs the enabled analyzers on every Nth frame, keeps the latest
result per capability in memory (served via REST/WebSocket) and periodically
persists results as ``Event`` rows.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any, Dict, Optional, Sequence

import cv2
import requests

from ingestion import RTSPCapture, VideoFileCapture
from ingestion.webcam import WebcamCapture
from inference.degradation_monitor import (
    CONDITION_BLURRY,
    CONDITION_LOW_LIGHT,
    CONDITION_NOISY,
    ConditionMonitor,
    DegradationReport,
)
from inference import BaseAnalyzer, get_analyzer

from .config import settings
from .database import SessionLocal
from .models import Event
from inference.face_association import FaceOverlayTracker

logger = logging.getLogger(__name__)


class FaceVerificationAnalyzer:
    """Adapter for the stateful ArcFace watchlist engine."""

    name = "face_verification"

    def __init__(self) -> None:
        from inference.face_verification import FaceVerificationEngine

        self.engine = FaceVerificationEngine(
            db_path=settings.watchlist_db_path,
            detector_backend="retinaface",
            model_name="ArcFace",
            draw=True,
        )
        self.annotated_frame = None

    def process(self, frame):
        results = self.engine.process_frame(frame, draw=True)
        self.annotated_frame = self.engine.annotated_frame
        return {"capability": self.name, "faces": results, "count": len(results)}


class ANPRAnalyzer:
    """Adapter for the stateful ANPR engine."""

    name = "anpr"

    def __init__(self) -> None:
        from inference.anpr import ANPREngine

        self.engine = ANPREngine()
        self.annotated_frame = None

    def process(self, frame):
        plates = self.engine.process_frame(frame)
        return {"capability": self.name, "plates": plates, "count": len(plates)}

    def close(self):
        self.engine.close()


def build_analyzer(name: str) -> BaseAnalyzer:
    """Create an analyzer, applying global settings where relevant."""
    kwargs: Dict[str, Any] = {}
    if name in ("detection", "tracking"):
        kwargs["model_name"] = settings.yolo_model
    if name == "face_verification":
        return FaceVerificationAnalyzer()  # type: ignore[return-value]
    if name == "anpr":
        return ANPRAnalyzer()  # type: ignore[return-value]
    return get_analyzer(name, **kwargs)


def create_capture(source_url: str):
    """Pick the right capture backend for a source URL."""
    if source_url.strip().isdigit():
        return WebcamCapture(
            int(source_url.strip()), reconnect_delay=settings.rtsp_reconnect_delay
        )
    if source_url.lower().startswith(("rtsp://", "rtsps://")):
        return RTSPCapture(source_url, reconnect_delay=settings.rtsp_reconnect_delay)
    return VideoFileCapture(source_url, loop=True)


class StreamPipeline:
    """Capture + inference worker for a single stream (one daemon thread)."""

    def __init__(self, stream_id: int, source_url: str, capabilities: Sequence[str]) -> None:
        self.stream_id = stream_id
        self.source_url = source_url
        self.capabilities = list(capabilities)
        self.capture = create_capture(source_url)
        self.analyzers: Dict[str, BaseAnalyzer] = {}
        for name in self.capabilities:
            try:
                self.analyzers[name] = build_analyzer(name)
            except Exception:
                # An optional capability (notably DeepFace) must not prevent
                # capture, tracking, or the live dashboard from starting.
                logger.exception(
                    "Stream %s: capability '%s' is unavailable and was skipped",
                    stream_id, name,
                )
        self._base_confidences = {
            name: float(getattr(analyzer, "confidence"))
            for name, analyzer in self.analyzers.items()
            if hasattr(analyzer, "confidence")
        }
        # Split analyzers by cost.  The main loop runs the light modules
        # (tracking) on EVERY processed frame so boxes track live, while the
        # slow modules (DeepFace face verification, ANPR OCR) run on a
        # dedicated worker thread that always works on the LATEST queued
        # frame only - a multi-second face pass can therefore never stall
        # tracking or the MJPEG overlay.
        self._heavy_names = [
            name for name in ("face_verification", "anpr") if name in self.analyzers
        ]
        self._light_names = [
            name for name in self.analyzers if name not in self._heavy_names
        ]
        self._heavy_queue: "queue.Queue" = queue.Queue(maxsize=1)
        self._heavy_thread: Optional[threading.Thread] = None
        self._heavy_frame_count = 0  # frame-skip counter for face verification
        self._persist_lock = threading.Lock()  # _maybe_persist runs on 2 threads
        # Face -> person association + live overlay state (see
        # inference.face_association): keeps the red face box following its
        # person between expensive ArcFace passes.
        self._face_overlay = FaceOverlayTracker(
            ttl_seconds=settings.face_state_ttl_seconds
        )
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._results_lock = threading.Lock()
        self._results: Dict[str, Dict[str, Any]] = {}
        self._latest_frame = None
        self._last_frame_id = 0
        self._frames_seen = 0
        self._next_persist_at: Dict[str, float] = {}
        self._zone_monitor = None
        self._alert_last: Dict[str, float] = {}
        self._last_observability_log = 0.0
        self._condition_monitor = ConditionMonitor(history_size=10)
        self._degradation = DegradationReport(
            condition="CLEAR", severity=0.0, raw_metrics={}
        )
        self._last_condition_check = 0.0
        self._last_condition_log = 0.0
        self._last_logged_condition = None
        if "tracking" in self.capabilities:
            logger.info("[FENCE] Zone monitor will initialize with the first frame size")

    # ------------------------------------------------------------- life-cycle
    def start(self) -> None:
        """Start the pipeline (no-op if already running)."""
        if self.is_running:
            return
        self._stop_event.clear()
        if not self.capture.is_running:
            self.capture.start()
        self._thread = threading.Thread(
            target=self._loop, name=f"ibvap-pipeline:{self.stream_id}", daemon=True
        )
        self._thread.start()
        if self._heavy_names and (
            self._heavy_thread is None or not self._heavy_thread.is_alive()
        ):
            self._heavy_thread = threading.Thread(
                target=self._heavy_loop,
                name=f"ibvap-heavy:{self.stream_id}",
                daemon=True,
            )
            self._heavy_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the pipeline thread and the underlying capture."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        # Wake the heavy worker and let it exit BEFORE analyzers are closed.
        try:
            self._heavy_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._heavy_thread is not None and self._heavy_thread.is_alive():
            self._heavy_thread.join(timeout=timeout)
        self._heavy_thread = None
        self.capture.stop(timeout=timeout)
        for analyzer in self.analyzers.values():
            close = getattr(analyzer, "close", None)
            if close:
                close()

    @property
    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    # ------------------------------------------------------------------ access
    def snapshot(self) -> Dict[str, Any]:
        """Latest result of every capability on this stream."""
        with self._results_lock:
            results = {name: dict(payload) for name, payload in self._results.items()}
        return {
            "stream_id": self.stream_id,
            "running": self.is_running,
            "frames_seen": self._frames_seen,
            "results": results,
            "frame_available": self._latest_frame is not None,
            "degradation": self._degradation.as_dict(),
            "adaptive": {
                "mode": (
                    "night_enhancement"
                    if settings.night_enhance_enabled
                    and (
                        self._degradation.condition == CONDITION_LOW_LIGHT
                        or self._degradation.raw_metrics.get("brightness", float("inf"))
                        < settings.brightness_threshold
                    )
                    else "standard"
                ),
                "reliability_score": round(max(0.0, 1.0 - self._degradation.severity), 4),
                "fence_consensus_frames": (
                    settings.degraded_consensus_frames
                    if self._is_degraded() else 2
                ),
            },
        }

    def latest_frame_jpeg(self) -> Optional[bytes]:
        """Return the current capture frame with the latest AI overlay.

        Capture owns the raw, continuously-updating frame buffer.  Inference
        may be much slower, so its latest tracks are drawn over whichever raw
        frame is current instead of publishing an old annotated image.  This
        is deliberately separate from ``_loop`` so model execution can never
        stall the browser video path.
        """
        captured = self.capture.latest()
        if captured is None:
            return None
        with self._results_lock:
            payloads = {name: dict(payload) for name, payload in self._results.items()}
            degradation = self._degradation
        frame = self._annotate_frame(captured.data, payloads, degradation)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return encoded.tobytes() if ok else None

    def run_once(self, capability: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Run one capability on the next available frame (on demand)."""
        analyzer = self.analyzers.get(capability)
        if analyzer is None:
            analyzer = build_analyzer(capability)
            self.analyzers[capability] = analyzer
        if not self.capture.is_running:
            self.capture.start()
        frame = self.capture.read(timeout=timeout)
        if frame is None:
            return None
        return self._run_analyzer(capability, analyzer, frame)

    # --------------------------------------------------------------- internals
    def _loop(self) -> None:
        logger.info("Pipeline started for stream %s (%s)", self.stream_id, self.capabilities)
        while not self._stop_event.is_set():
            frame = self.capture.read(timeout=0.5, after_id=self._last_frame_id)
            if frame is None:
                continue
            self._last_frame_id = frame.frame_id
            self._frames_seen += 1
            now = time.monotonic()
            if now - self._last_condition_check >= settings.condition_check_interval_seconds:
                self._degradation = self._condition_monitor.process_frame(frame.data)
                self._last_condition_check = now
                if (
                    self._degradation.condition != self._last_logged_condition
                    or now - self._last_condition_log >= 30.0
                ):
                    logger.info(
                        "[CONDITION] %s | severity=%.2f metrics=%s",
                        self._degradation.condition,
                        self._degradation.severity,
                        self._degradation.raw_metrics,
                    )
                    self._last_logged_condition = self._degradation.condition
                    self._last_condition_log = now
            with self._results_lock:
                self._latest_frame = frame.data.copy()
            if self._frames_seen == 1 or self._frames_seen % 100 == 0:
                logger.info(
                    "[STREAM] Frame received: stream=%s count=%s condition=%s severity=%.2f",
                    self.stream_id,
                    self._frames_seen,
                    self._degradation.condition,
                    self._degradation.severity,
                )
            # Analyze every Nth frame to keep CPU usage sane.
            if self._frames_seen % settings.process_every_n_frames:
                continue
            working_frame = frame.data
            if "tracking" in self.capabilities and self._zone_monitor is None:
                try:
                    from inference.virtual_fence import ZoneMonitor

                    height, width = working_frame.shape[:2]
                    self._zone_monitor = ZoneMonitor(
                        [(0.1 * width, 0.35 * height), (0.9 * width, 0.35 * height),
                         (0.9 * width, 0.95 * height), (0.1 * width, 0.95 * height)],
                        zone_name="restricted", inside_threshold=2,
                    )
                    logger.info("[FENCE] Zone monitor ready for %sx%s", width, height)
                except Exception:
                    logger.exception("[FENCE] failed to initialize zone monitor")
            degraded = self._is_degraded()
            low_light = (
                self._degradation.condition == CONDITION_LOW_LIGHT
                or self._degradation.raw_metrics.get("brightness", float("inf"))
                < settings.brightness_threshold
            )
            if settings.night_enhance_enabled and low_light:
                try:
                    from inference.night_enhance import enhance_frame

                    working_frame, method = enhance_frame(
                        working_frame,
                        brightness_threshold=settings.brightness_threshold,
                    )
                    if method != "passthrough":
                        logger.info("[NIGHT] Enhancement activated: %s", method)
                except Exception:
                    logger.exception("[NIGHT] enhancement failed; using original frame")
            for name in self._light_names:
                analyzer = self.analyzers.get(name)
                if hasattr(analyzer, "confidence") and name in self._base_confidences:
                    analyzer.confidence = (
                        max(0.1, self._base_confidences[name] * 0.65)
                        if degraded
                        else self._base_confidences[name]
                    )
            with self._results_lock:
                self._latest_frame = working_frame.copy()
            payloads = {}
            # Light modules run on EVERY processed frame -> tracking boxes
            # refresh live.  The heavy modules are offered the newest frame
            # and run on their own thread (see _heavy_loop).
            for name in self._light_names:
                analyzer = self.analyzers.get(name)
                if analyzer is None:
                    continue
                # When tracking is stored, re-anchor and publish the face
                # overlay in the SAME critical section (atomic pair).
                on_stored = self._publish_face_overlay if name == "tracking" else None
                payload = self._run_analyzer(
                    name, analyzer, frame, working_frame, on_stored=on_stored
                )
                payloads[name] = payload
                self._handle_module_events(name, payload, working_frame)
            self._offer_heavy(frame, working_frame)
            with self._results_lock:
                self._latest_frame = self._annotate_frame(working_frame, payloads)
        logger.info("Pipeline stopped for stream %s", self.stream_id)

    def _publish_face_overlay(self) -> None:
        """Re-anchor verified faces to the just-stored tracking result.

        Caller MUST already hold ``self._results_lock``: this runs inside
        the ``on_stored`` hook of ``_run_analyzer``, in the same critical
        section that stores the tracking/face payload.  The dashboard's
        atomic snapshot therefore always contains a consistent pair - the
        person box and its re-anchored face box come from the exact same
        sweep, and the red face box can never visibly lag its person.
        """
        face_payload = self._results.get("face_verification")
        if face_payload is None:
            return  # no face verification result yet - nothing to overlay
        tracks = (self._results.get("tracking") or {}).get("tracks", [])
        faces = self._face_overlay.live_faces(tracks, time.time())
        face_payload["faces"] = faces
        face_payload["count"] = len(faces)

    # ------------------------------------------------------- heavy worker
    def _offer_heavy(self, frame, working_frame) -> None:
        """Offer the newest frame to the heavy worker, dropping stale ones.

        The queue holds a single slot: if the worker is still busy with the
        previous frame, that older frame is REPLACED so the worker always
        resumes analysis on the freshest image - the same latest-frame policy
        the capture buffer applies to ``cap.read()``.  This is what keeps the
        tracking overlay live while a multi-second face pass is in flight.
        """
        if not self._heavy_names:
            return
        item = (frame, working_frame)
        try:
            self._heavy_queue.put_nowait(item)
        except queue.Full:
            try:
                self._heavy_queue.get_nowait()
                self._heavy_queue.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def _heavy_loop(self) -> None:
        """Run the slow modules (face verification / ANPR) off the main loop.

        Consumes only the LATEST queued frame, so tracking on the main thread
        keeps refreshing at full speed while DeepFace takes its time.  Face
        verification - the heaviest module - additionally runs only on every
        ``settings.face_every_n_frames``-th frame offered to this worker.
        """
        logger.info(
            "[stream %s] heavy worker started (%s), faces every %d frame(s)",
            self.stream_id, ", ".join(self._heavy_names),
            settings.face_every_n_frames,
        )
        while not self._stop_event.is_set():
            try:
                item = self._heavy_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            frame, working_frame = item
            self._heavy_frame_count += 1
            degraded = self._is_degraded()
            for name in self._heavy_names:
                if self._stop_event.is_set():
                    break
                analyzer = self.analyzers.get(name)
                if analyzer is None:
                    continue
                # frame-skip: face verification runs every Nth offered frame
                if name == "face_verification" and (
                    self._heavy_frame_count % max(1, settings.face_every_n_frames)
                ):
                    continue
                if hasattr(analyzer, "confidence") and name in self._base_confidences:
                    analyzer.confidence = (
                        max(0.1, self._base_confidences[name] * 0.65)
                        if degraded
                        else self._base_confidences[name]
                    )
                if name == "face_verification":
                    # Associate verified faces with the CURRENT ByteTrack
                    # persons and anchor each face inside its person box.
                    # Runs BEFORE storage (on_result), so the published
                    # payload never shows un-associated faces.
                    def associate(result: Dict[str, Any]) -> None:
                        with self._results_lock:
                            tracks = (self._results.get("tracking") or {}).get("tracks", [])
                        self._face_overlay.update_verified(
                            result.get("faces", []), tracks
                        )

                    on_result = associate
                else:
                    on_result = None
                # After the face payload is stored, re-anchor and publish the
                # live overlay INSIDE the same storage critical section, so
                # consumers (dashboard snapshot) never catch the raw verified
                # bbox before it is re-projected onto the current person box.
                on_stored = (
                    self._publish_face_overlay if name == "face_verification" else None
                )
                payload = self._run_analyzer(
                    name, analyzer, frame, working_frame,
                    on_result=on_result, on_stored=on_stored,
                )
                self._handle_module_events(name, payload, working_frame)
        logger.info("[stream %s] heavy worker stopped", self.stream_id)

    def _run_analyzer(self, name: str, analyzer: BaseAnalyzer, frame, image=None,
                      on_result=None, on_stored=None) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            result = analyzer.process(image if image is not None else frame.data)
        except Exception as exc:  # one failing capability must not kill the pipeline
            logger.exception("Analyzer '%s' failed on stream %s", name, self.stream_id)
            result = {"capability": name, "error": str(exc)}
        if on_result is not None:
            # Pre-storage hook (e.g. face -> person association) - runs BEFORE
            # the result becomes visible in _results, so consumers never see
            # an un-associated face payload.
            on_result(result)
        payload = {
            **result,
            "capability": name,
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        with self._results_lock:
            self._results[name] = payload
            annotated = getattr(analyzer, "annotated_frame", None)
            if annotated is not None:
                self._latest_frame = annotated.copy()
            if on_stored is not None:
                # Optional atomic follow-up (e.g. re-anchoring the face
                # overlay to this exact tracking result) - runs inside the
                # same lock, so consumers never see a half-updated pair.
                on_stored()
        self._maybe_persist(name, payload)
        if name in ("tracking", "face_verification", "anpr"):
            now = time.monotonic()
            if now - self._last_observability_log >= 2.0:
                logger.info(
                    "[%s] frame=%s objects=%s latency=%.0fms",
                    name.upper(), frame.frame_id, payload.get("count", 0), payload["latency_ms"],
                )
                self._last_observability_log = now
        return payload

    def _handle_module_events(self, name: str, payload: Dict[str, Any], image) -> None:
        if name == "tracking" and self._zone_monitor is not None:
            # ZoneMonitor owns the per-track consecutive-frame state. Raising
            # its threshold only while the measured scene is degraded gives a
            # genuine negative/positive consensus guard without duplicating
            # fence state or emitting fabricated alerts.
            self._zone_monitor.inside_threshold = (
                settings.degraded_consensus_frames if self._is_degraded() else 2
            )
            events = self._zone_monitor.update(payload.get("tracks", []), payload.get("timestamp"))
            for event in events:
                logger.info("[FENCE] Intrusion detected: %s", event)
                self._emit_alert(
                    module="fence", severity="critical",
                    message=f"{event['class']}#{event['track_id']} entered zone '{event['zone_name']}'",
                    track_id=event["track_id"], timestamp=event["timestamp"], image=image,
                )
        elif name == "face_verification":
            for face in payload.get("faces", []):
                if face.get("name") == "UNKNOWN":
                    continue
                logger.info("[FACE] Watchlist match: %s/%.3f", face["name"], face.get("confidence", 0.0))
                self._emit_alert(
                    module="face_verification", severity="warning",
                    message=f"watchlist face '{face['name']}' matched (conf {face.get('confidence', 0.0):.2f})",
                    track_id=None, timestamp=time.time(), image=image,
                    dedupe_key=f"face:{face['name']}",
                )
        elif name == "anpr":
            for plate in payload.get("plates", []):
                logger.info("[ANPR] Plate detected: %s", plate.get("plate_text"))
                self._emit_alert(
                    module="anpr", severity="info",
                    message=f"plate {plate['plate_text']} read (conf {plate.get('confidence', 0.0):.2f})",
                    track_id=None, timestamp=time.time(), image=image,
                    dedupe_key=f"plate:{plate['plate_text']}",
                )

    def _emit_alert(self, *, module, severity, message, track_id, timestamp, image, dedupe_key=None):
        key = dedupe_key or f"{module}:{track_id}:{message}"
        now = time.time()
        if now - self._alert_last.get(key, 0.0) < 10.0:
            return
        self._alert_last[key] = now
        thumbnail = None
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ok:
            import base64

            thumbnail = base64.b64encode(encoded.tobytes()).decode("ascii")
        try:
            # Reliability is derived from the active, measured degradation
            # report.  It is not a synthetic detector confidence: it tells
            # alert consumers how trustworthy the visual conditions were.
            reliability = max(0.0, min(1.0, 1.0 - self._degradation.severity))
            response = requests.post(
                settings.alert_ingest_url,
                json={"module": module, "severity": severity, "message": message,
                      "track_id": track_id, "timestamp": timestamp,
                      "camera_id": str(self.stream_id), "thumbnail_base64": thumbnail,
                      "detection_condition": self._degradation.condition,
                      "detection_reliability_score": round(reliability, 4)},
                timeout=5,
            )
            response.raise_for_status()
            logger.info("[ALERT] Alert emitted: %s", message)
        except requests.RequestException:
            logger.exception("[ALERT] Failed to emit alert")

    def _is_degraded(self) -> bool:
        return self._degradation.condition in (CONDITION_BLURRY, CONDITION_NOISY) or (
            self._degradation.severity > 0.6
        )

    def _annotate_frame(self, frame, payloads: Dict[str, Dict[str, Any]], degradation=None):
        """Render boxes from actual module results onto the published frame."""
        scene = frame.copy()
        tracking = payloads.get("tracking", {})
        for track in tracking.get("tracks", []):
            x1, y1, x2, y2 = (int(value) for value in track["bbox"])
            inside = self._zone_monitor is not None and self._zone_monitor.is_inside(track["bbox"])
            color = (0, 0, 255) if inside else (0, 220, 0)
            label = f"{track['class'].upper()}  ID: {track.get('track_id')}  {track.get('confidence', 0.0) * 100:.0f}%"
            cv2.rectangle(scene, (x1, y1), (x2, y2), color, 2)
            cv2.putText(scene, label, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        for face in payloads.get("face_verification", {}).get("faces", []):
            x1, y1, x2, y2 = (int(value) for value in face["bbox"])
            color = (0, 200, 0) if face.get("name") != "UNKNOWN" else (0, 0, 220)
            label = f"FACE: {face.get('name', 'UNKNOWN')}  {face.get('confidence', 0.0) * 100:.0f}%"
            cv2.rectangle(scene, (x1, y1), (x2, y2), color, 2)
            cv2.putText(scene, label, (x1, min(y2 + 20, scene.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        if self._zone_monitor is not None:
            self._zone_monitor.draw_polygon(scene)
        report = degradation or self._degradation
        color = (0, 200, 0) if report.condition == "CLEAR" else (0, 165, 255)
        cv2.putText(
            scene, f"{report.condition} | {report.severity:.0%}", (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
        return scene

    def _maybe_persist(self, name: str, payload: Dict[str, Any]) -> None:
        """Persist the latest result as an Event row every N seconds."""
        now = time.monotonic()
        if now < self._next_persist_at.get(name, 0.0):
            return
        self._next_persist_at[name] = now + settings.persist_interval_seconds
        # Runs on both the main loop and the heavy worker -> serialize.
        with self._persist_lock:
            try:
                db = SessionLocal()
                try:
                    db.add(
                        Event(
                            stream_id=self.stream_id,
                            capability=name,
                            payload=json.loads(json.dumps(payload, default=str)),
                        )
                    )
                    db.commit()
                finally:
                    db.close()
            except Exception:
                logger.exception("Failed to persist event (stream %s, %s)", self.stream_id, name)

