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
