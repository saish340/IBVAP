"""Local webcam capture backend."""

from __future__ import annotations

import cv2

from .base import BaseVideoCapture


class WebcamCapture(BaseVideoCapture):
    """Capture from a local camera index such as 0 or 1."""

    def __init__(self, camera_index: int = 0, **kwargs) -> None:
        self.camera_index = int(camera_index)
        super().__init__(source=str(self.camera_index), name=f"webcam-{self.camera_index}", **kwargs)

    def _create_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)