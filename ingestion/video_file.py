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

    def _create_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self.source)
