"""Shared data structures and the base class for all video capture backends.

Also hosts :class:`LatestFrameBuffer` - the lock-protected latest-frame
buffer that decouples capture from inference (see the class docstring).
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
            self._after_frame()
        self._release()

    def _after_frame(self) -> None:
        """Optional source-specific pacing hook after publishing a frame."""

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


#: pause between retries while the source stops delivering frames (the
#: alternative - a truly tight retry loop - would pin a CPU core)
ERROR_DELAY = 0.05


class LatestFrameBuffer:
    """Continuously grab frames on a daemon thread, keeping only the newest.

    Reading an RTSP/webcam frame and running inference on it in the same loop
    makes the displayed image stale by the time results arrive: every
    expensive analysis pass blocks the next ``cap.read()``.  This class fixes
    that by owning the camera on a dedicated daemon thread that calls
    ``cap.read()`` in a tight loop and overwrites a **single** slot under a
    :class:`threading.Lock`; every older frame is discarded the instant a
    newer one arrives (deliberately *not* a queue, so a slow consumer can
    never build a backlog).  Consumers call :meth:`get_latest` and receive
    the freshest frame instantly, without ever blocking on camera I/O.

    ``on_frame`` is invoked with each raw frame straight from the reader
    thread, which lets a display or stream path stay live even while
    inference blocks another thread.

    Example::

        buffer = LatestFrameBuffer(cap, name="ibvap-frames:cam-01")
        buffer.start()
        while running:
            ok, frame = buffer.get_latest()   # instant, never blocks on I/O
    """

    def __init__(
        self,
        cap: Any,
        name: str = "ibvap-frame-buffer",
        on_frame: Optional[Callable[[np.ndarray], None]] = None,
        error_delay: float = ERROR_DELAY,
    ) -> None:
        self.cap = cap
        self.name = name
        self.on_frame = on_frame
        self.error_delay = error_delay
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[np.ndarray] = None
        self._frame_id = 0
        self._running = False
        self._read_failures = 0

    # -------------------------------------------------------------- life-cycle
    def start(self) -> None:
        """Start the background reader thread (no-op if already running)."""
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the reader thread (does *not* release the underlying capture)."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ access
    def get_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return ``(ok, frame)`` for the most recent frame, instantly.

        Never touches the camera - the frame is served from the single
        latest slot, so a caller can never be stalled by a blocking
        ``cap.read()``.  ``ok`` is ``False`` until the first frame arrives.
        """
        with self._lock:
            frame = self._latest
        return (frame is not None, frame)

    @property
    def frame_id(self) -> int:
        """Monotonic id of the stored frame (increments on every new grab)."""
        with self._lock:
            return self._frame_id

    # --------------------------------------------------------------- internals
    def _loop(self) -> None:
        """Reader loop: grab -> overwrite the latest slot -> repeat."""
        while not self._stop_event.is_set():
            try:
                ok, frame = self.cap.read()
            except Exception as exc:
                ok, frame = False, None
                logger.debug("[%s] capture read raised: %s", self.name, exc)
            if not ok or frame is None:
                # Transient failure: keep the last good frame available and
                # back off briefly instead of spinning on a dead source.
                self._read_failures += 1
                if self._read_failures == 1 or self._read_failures % 50 == 0:
                    logger.warning(
                        "[%s] frame read failing (attempt %d)",
                        self.name, self._read_failures,
                    )
                if self._stop_event.wait(self.error_delay):
                    break
                continue
            self._read_failures = 0
            with self._lock:
                self._frame_id += 1
                self._latest = frame  # overwrite: older frames are discarded
            if self.on_frame is not None:
                # Called outside the lock so a slow display/stream hook can
                # never delay the next grab.
                try:
                    self.on_frame(frame)
                except Exception:
                    logger.exception("[%s] on_frame callback failed", self.name)
        logger.info("[%s] reader thread stopped", self.name)
