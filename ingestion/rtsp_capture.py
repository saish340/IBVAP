"""RTSP / RTSPS network stream capture with automatic reconnection."""

from __future__ import annotations

import os
from typing import Optional

import cv2

from .base import BaseVideoCapture


class RTSPCapture(BaseVideoCapture):
    """Capture frames from an RTSP URL, reconnecting automatically on failure.

    Example:
        capture = RTSPCapture("rtsp://user:pass@camera.local:554/stream1")
        capture.start()
        frame = capture.read(timeout=5.0)
    """

    def __init__(
        self,
        rtsp_url: str,
        name: Optional[str] = None,
        reconnect_delay: float = 2.0,
        transport: str = "tcp",
    ) -> None:
        super().__init__(source=rtsp_url, name=name or rtsp_url, reconnect_delay=reconnect_delay)
        self.transport = transport

    def _create_capture(self) -> cv2.VideoCapture:
        # Ask FFmpeg (OpenCV's RTSP backend) to use a reliable transport and
        # to time out instead of hanging forever. This must be set before the
        # VideoCapture opens; options are `key;value` pairs joined by `|`.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;{self.transport}|stimeout;5000000",
        )
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.grab()  # discard the stale buffered frame
        return cap
