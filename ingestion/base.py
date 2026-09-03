"""Shared data structures and the base class for all video capture backends."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class Frame:
    """A single video frame with capture metadata."""

    data: np.ndarray  # BGR image, as produced by OpenCV
    frame_id: int
    timestamp: float  # unix epoch seconds
    source: str

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])


class BaseVideoCapture(ABC):
    """Continuously reads frames on a daemon thread.

    Only the most recent frame is retained, so consumers always receive the
    freshest image and slow downstream processing never builds a backlog.
    Subclasses implement :meth:`_create_capture` to build their
    ``cv2.VideoCapture`` instance; the base class handles threading,
    reconnection and the latest-frame buffer.
    """

    def __init__(
        self,
        source: str,
        name: Optional[str] = None,
        reconnect_delay: float = 2.0,
        retry_delay: Optional[float] = None,
    ) -> None:
        self.source = source
        self.name = name or str(source)
        # Delay before retrying after a failed (re)open or read error.
        self.retry_delay = retry_delay if retry_delay is not None else reconnect_delay
        self._cap: Any = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[Frame] = None
        self._frame_id = 0
        self._running = False
        # When True a failed read ends capture instead of retrying (used for
        # finite sources such as non-looping video files).
        self._stop_on_read_failure = False

    # ------------------------------------------------------------- life-cycle
    def start(self) -> None:
        """Start the background capture thread (no-op if already running)."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name=f"ibvap-capture:{self.name}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the capture thread and release the underlying device."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        self._running = False
        self._release()

    @property
    def is_running(self) -> bool:
        return self._running

    # ----------------------------------------------------------------- access
    def latest(self) -> Optional[Frame]:
        """Return the most recent frame, or ``None`` if none captured yet."""
        with self._lock:
            return self._latest

    def read(self, timeout: float = 1.0, after_id: Optional[int] = None) -> Optional[Frame]:
        """Wait up to ``timeout`` seconds for a frame newer than ``after_id``.

        Returns ``None`` on timeout or when the capture has been stopped.
        Pass the ``frame_id`` of the previously processed frame to avoid
        receiving the same image twice.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                frame = self._latest
            if frame is not None and (after_id is None or frame.frame_id > after_id):
                return frame
            if self._stop_event.is_set():
                return None
            time.sleep(0.02)
        return None

    # -------------------------------------------------------------- internals
    @abstractmethod
    def _create_capture(self) -> Any:
        """Create and return an opened ``cv2.VideoCapture`` (or equivalent)."""

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._is_open():
                if not self._open():
                    time.sleep(self.retry_delay)
                    continue
            ok, data = self._cap.read()
            if not ok:
                self._release()
                if self._stop_on_read_failure:
                    self._stop_event.set()
                    break
                time.sleep(self.retry_delay)
                continue
            self._frame_id += 1
            frame = Frame(
                data=data, frame_id=self._frame_id, timestamp=time.time(), source=self.source
            )
            with self._lock:
                self._latest = frame
        self._release()

    def _open(self) -> bool:
        try:
            cap = self._create_capture()
        except Exception:
            return False
        self._cap = cap if cap is not None and cap.isOpened() else None
        return self._cap is not None

    def _is_open(self) -> bool:
        try:
            return self._cap is not None and self._cap.isOpened()
        except Exception:
            return False

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
