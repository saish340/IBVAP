"""Compatibility shim: :class:`LatestFrameBuffer` moved to ``ingestion.base``.

The implementation now lives next to the other capture primitives in
:class:`ingestion.base.BaseVideoCapture`; this module re-exports it so the
historical ``from ingestion.frame_buffer import LatestFrameBuffer`` import
keeps working.
"""

from .base import ERROR_DELAY, LatestFrameBuffer

__all__ = ["ERROR_DELAY", "LatestFrameBuffer"]

