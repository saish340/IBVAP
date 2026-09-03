"""Registry for managing multiple concurrent video captures."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from .base import BaseVideoCapture
from .rtsp_capture import RTSPCapture
from .video_file import VideoFileCapture


class StreamManager:
    """Create, look up and stop video captures keyed by an arbitrary id."""

    def __init__(self) -> None:
        self._streams: Dict[str, BaseVideoCapture] = {}
        self._lock = threading.Lock()

    def add(
        self, stream_id: str, source_url: str, start: bool = True, **kwargs: Any
    ) -> BaseVideoCapture:
        """Register a stream under ``stream_id``.

        RTSP/RTSPS URLs get an :class:`RTSPCapture`; anything else (file
        paths, ``http(s)://`` video URLs) gets a looping
        :class:`VideoFileCapture`. Extra keyword arguments are forwarded to
        the capture class (e.g. ``transport=`` or ``loop=``).
        """
        if str(source_url).lower().startswith(("rtsp://", "rtsps://")):
            capture: BaseVideoCapture = RTSPCapture(source_url, **kwargs)
        else:
            capture = VideoFileCapture(source_url, **kwargs)
        if start:
            capture.start()
        with self._lock:
            old = self._streams.get(stream_id)
            self._streams[stream_id] = capture
        if old is not None:
            old.stop()
        return capture

    def get(self, stream_id: str) -> Optional[BaseVideoCapture]:
        with self._lock:
            return self._streams.get(stream_id)

    def remove(self, stream_id: str) -> bool:
        """Stop and remove a stream. Returns ``True`` if it existed."""
        with self._lock:
            capture = self._streams.pop(stream_id, None)
        if capture is None:
            return False
        capture.stop()
        return True

    def list_ids(self) -> List[str]:
        with self._lock:
            return sorted(self._streams)

    def stop_all(self) -> None:
        with self._lock:
            captures = list(self._streams.values())
            self._streams.clear()
        for capture in captures:
            capture.stop()
