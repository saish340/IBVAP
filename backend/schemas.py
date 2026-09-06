"""Pydantic request/response schemas for the API."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class StreamCreate(BaseModel):
    """Body for ``POST /streams``."""

    name: str = Field(min_length=1, max_length=120, examples=["front-door-cam"])
    source_url: str = Field(
        min_length=1,
        max_length=500,
        examples=["rtsp://user:pass@camera.local:554/stream1"],
    )
    #: inference capabilities to attach to this stream
    capabilities: List[str] = Field(default_factory=lambda: ["detection"])
    #: start analyzing immediately after creation
    auto_start: bool = True


class StreamOut(BaseModel):
    """Response model for a stream."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    source_url: str
    capabilities: List[str]
    enabled: bool
    running: bool = False


class EventOut(BaseModel):
    """Response model for a persisted analysis event."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    stream_id: Optional[int]
    capability: str
    payload: Dict[str, Any]
    created_at: dt.datetime


# ------------------------------------------------------------------- Alerts

class AlertCreate(BaseModel):
    """Body for ``POST /events/ingest``."""

    module: str = Field(min_length=1, max_length=64, examples=["fence"])
    severity: str = Field(
        default="info", max_length=16, examples=["warning"]
    )
    message: str = Field(
        min_length=1, max_length=1000,
        examples=["Person#3 entered the restricted zone"],
    )
    track_id: Optional[int] = Field(default=None, examples=[7])
    timestamp: Optional[float] = Field(
        default=None, description="unix epoch seconds; defaults to server time"
    )
    camera_id: Optional[str] = Field(default=None, max_length=120, examples=["cam-01"])
    thumbnail_base64: Optional[str] = Field(
        default=None, description="base64-encoded JPEG/PNG frame thumbnail"
    )
    detection_condition: Optional[str] = Field(default=None, max_length=32)
    detection_reliability_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class AlertOut(BaseModel):
    """Response model for an alert event."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    module: str
    severity: str
    message: str
    track_id: Optional[int]
    timestamp: Optional[float]
    camera_id: Optional[str]
    thumbnail_base64: Optional[str]
    detection_condition: Optional[str]
    detection_reliability_score: Optional[float]
    created_at: dt.datetime


class AlertHistoryOut(BaseModel):
    """Paginated past-alert response."""

    items: List[AlertOut]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------- Watchlist

class WatchlistEnrollRequest(BaseModel):
    """Body for ``POST /watchlist/enroll``."""

    name: str = Field(min_length=1, max_length=120, examples=["Alice"])
    image_base64: str = Field(
        min_length=10,
        description="base64-encoded photo (JPG/PNG) with the face; "
                    "a 'data:image/...;base64,' prefix is accepted too",
        examples=["iVBORw0KGgo..."],
    )


class WatchlistEntryOut(BaseModel):
    """A single enrolled watchlist entry (one per name)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    model: str
    enrolled_at: str


class WatchlistEnrollResponse(BaseModel):
    """Confirmation of a new enrollment."""

    name: str
    bbox: List[int]
    face_confidence: float
    faces_in_image: int
    message: str
