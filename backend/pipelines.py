"""Per-stream processing pipelines: glue between ingestion and inference.

A :class:`StreamPipeline` reads frames from a capture backend on one worker
thread, runs the enabled analyzers on every Nth frame, keeps the latest
result per capability in memory (served via REST/WebSocket) and periodically
persists results as ``Event`` rows.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Sequence

from ingestion import RTSPCapture, VideoFileCapture
from inference import BaseAnalyzer, get_analyzer

from .config import settings
from .database import SessionLocal
from .models import Event

logger = logging.getLogger(__name__)


def build_analyzer(name: str) -> BaseAnalyzer:
    """Create an analyzer, applying global settings where relevant."""
    kwargs: Dict[str, Any] = {}
    if name in ("detection", "tracking"):
        kwargs["model_name"] = settings.yolo_model
    return get_analyzer(name, **kwargs)


def create_capture(source_url: str):
    """Pick the right capture backend for a source URL."""
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
            except ValueError:
                logger.warning("Stream %s: unknown capability '%s' skipped", stream_id, name)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._results_lock = threading.Lock()
        self._results: Dict[str, Dict[str, Any]] = {}
        self._last_frame_id = 0
        self._frames_seen = 0
        self._next_persist_at: Dict[str, float] = {}

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

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the pipeline thread and the underlying capture."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        self.capture.stop(timeout=timeout)

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
        }

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
            # Analyze every Nth frame to keep CPU usage sane.
            if self._frames_seen % settings.process_every_n_frames:
                continue
            for name, analyzer in list(self.analyzers.items()):
                self._run_analyzer(name, analyzer, frame)
        logger.info("Pipeline stopped for stream %s", self.stream_id)

    def _run_analyzer(self, name: str, analyzer: BaseAnalyzer, frame) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            result = analyzer.process(frame.data)
        except Exception as exc:  # one failing capability must not kill the pipeline
            logger.exception("Analyzer '%s' failed on stream %s", name, self.stream_id)
            result = {"capability": name, "error": str(exc)}
        payload = {
            **result,
            "capability": name,
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        with self._results_lock:
            self._results[name] = payload
        self._maybe_persist(name, payload)
        return payload

    def _maybe_persist(self, name: str, payload: Dict[str, Any]) -> None:
        """Persist the latest result as an Event row every N seconds."""
        now = time.monotonic()
        if now < self._next_persist_at.get(name, 0.0):
            return
        self._next_persist_at[name] = now + settings.persist_interval_seconds
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

