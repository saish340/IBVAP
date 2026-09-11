"""Lock-protected latest-frame buffer: capture decoupled from inference.

Reading an RTSP/webcam frame and running inference on it in the same loop
makes the displayed image stale by the time results arrive: every expensive
analysis pass blocks the next ``cap.read()``.  :class:`LatestFrameBuffer`
fixes that by owning the camera on a dedicated daemon thread that only ever
keeps the newest frame; consumers call :meth:`LatestFrameBuffer.get_latest`
and receive it instantly, without ever blocking on camera I/O.

The single-slot, overwrite-on-arrival policy mirrors
:class:`ingestion.base.BaseVideoCapture`, exposed as a lightweight wrapper
around an already-open ``cv2.VideoCapture`` so the end-to-end pipeline can
drop it in place of its direct ``cap.read()`` loop.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: pause between retries while the source stops delivering frames (the
#: alternative - a truly tight retry loop - would pin a CPU core)
ERROR_DELAY = 0.05


class LatestFrameBuffer:
    """Continuously grab frames on a daemon thread, keeping only the newest.

    The reader thread calls ``cap.read()`` in a tight loop and overwrites a
    single slot under a :class:`threading.Lock`; every older frame is
    discarded the instant a newer one arrives (deliberately *not* a queue, so
    a slow consumer can never build a backlog).  ``on_frame`` is invoked with
    each raw frame straight from the reader thread, which lets a display or
    stream path stay live even while inference blocks another thread.

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
        self._thread = threading.Thread(
            target=self._loop, name=self.name, daemon=True
        )
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
