"""IBVAP video ingestion.

Threaded capture back-ends that keep the freshest frame available for the
inference layer:

* :class:`RTSPCapture`      - RTSP/RTSPS IP-camera streams, auto-reconnect
* :class:`VideoFileCapture` - local files / HTTP video URLs (loops by default)
* :class:`StreamManager`    - registry for running many captures at once
"""

from .base import BaseVideoCapture, Frame, LatestFrameBuffer
from .rtsp_capture import RTSPCapture
from .stream_manager import StreamManager
from .video_file import VideoFileCapture

__all__ = [
    "BaseVideoCapture",
    "Frame",
    "LatestFrameBuffer",
    "RTSPCapture",
    "StreamManager",
    "VideoFileCapture",
]
