"""Local video file (or HTTP video URL) capture, optionally looping."""

from __future__ import annotations

from typing import Optional

import cv2

from .base import BaseVideoCapture


class VideoFileCapture(BaseVideoCapture):
    """Capture frames from a video file, replaying it in a loop by default.

    Useful for demos and testing when no RTSP camera is available.
    """

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        loop: bool = True,
        retry_delay: float = 0.1,
    ) -> None:
        super().__init__(source=str(path), name=name or str(path), retry_delay=retry_delay)
        self.loop = loop
        self._stop_on_read_failure = not loop
        self._frame_interval = 0.0

    def _create_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.source)
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
        # File input is a demo camera, so preserve its native rate. Without
        # pacing it runs flat-out, skips most frames for inference, burns CPU,
        # and makes ByteTrack IDs appear unstable.
        self._frame_interval = 1.0 / fps if fps > 0 else 0.0
        return capture

    def _after_frame(self) -> None:
        if self._frame_interval > 0:
            self._stop_event.wait(self._frame_interval)
