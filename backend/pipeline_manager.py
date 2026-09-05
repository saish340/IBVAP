"""Registry that keeps one :class:`StreamPipeline` per active stream."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Sequence

from .pipelines import StreamPipeline

logger = logging.getLogger(__name__)


class PipelineManager:
    """Create, look up and stop pipelines for streams (thread-safe)."""

    def __init__(self) -> None:
        self._pipelines: Dict[int, StreamPipeline] = {}
        self._lock = threading.Lock()

    def start(self, stream_id: int, source_url: str, capabilities: Sequence[str]) -> None:
        """(Re)create and start the pipeline for a stream."""
        with self._lock:
            old = self._pipelines.get(stream_id)
            pipeline = StreamPipeline(stream_id, source_url, capabilities)
            self._pipelines[stream_id] = pipeline
        if old is not None:
            old.stop()
        pipeline.start()

    def stop(self, stream_id: int) -> bool:
        with self._lock:
            pipeline = self._pipelines.pop(stream_id, None)
        if pipeline is None:
            return False
        pipeline.stop()
        return True

    def stop_all(self) -> None:
        with self._lock:
            pipelines = list(self._pipelines.values())
            self._pipelines.clear()
        for pipeline in pipelines:
            pipeline.stop()

    def is_running(self, stream_id: int) -> bool:
        with self._lock:
            pipeline = self._pipelines.get(stream_id)
        return bool(pipeline is not None and pipeline.is_running)

    def snapshot(self, stream_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            pipeline = self._pipelines.get(stream_id)
        return pipeline.snapshot() if pipeline is not None else None

    def get(self, stream_id: int) -> Optional[StreamPipeline]:
        """Return an active pipeline for direct frame access."""
        with self._lock:
            return self._pipelines.get(stream_id)

    def get_or_create(
        self, stream_id: int, source_url: str, capabilities: Sequence[str]
    ) -> StreamPipeline:
        """Return the pipeline for a stream, creating + starting it if needed."""
        with self._lock:
            pipeline = self._pipelines.get(stream_id)
            if pipeline is None:
                pipeline = StreamPipeline(stream_id, source_url, capabilities)
                self._pipelines[stream_id] = pipeline
        if not pipeline.is_running:
            pipeline.start()
        return pipeline

    def run_once(
        self, stream_id: int, source_url: str, capabilities: Sequence[str], capability: str
    ) -> Optional[Dict[str, Any]]:
        pipeline = self.get_or_create(stream_id, source_url, capabilities)
        return pipeline.run_once(capability)

    def autostart_all(self, streams) -> int:
        """Start pipelines for ``[(stream_id, source_url, capabilities), ...]``."""
        started = 0
        for stream_id, source_url, capabilities in streams:
            try:
                self.start(stream_id, source_url, capabilities)
                started += 1
            except Exception:
                logger.exception("Failed to autostart stream %s", stream_id)
        return started


#: Shared singleton used by the FastAPI app.
pipeline_manager = PipelineManager()
