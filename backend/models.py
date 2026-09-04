"""SQLAlchemy ORM models."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Stream(Base):
    """A video source (RTSP camera or video file) registered for analysis."""

    __tablename__ = "streams"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    source_url: Mapped[str] = mapped_column(String(500))
    #: names of inference capabilities to run, e.g. ["detection", "ocr"]
    capabilities: Mapped[list] = mapped_column(JSON, default=list)
    #: enabled streams are (re)started automatically on backend boot
    enabled: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    events: Mapped[list["Event"]] = relationship(
        back_populates="stream", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Stream id={self.id} name={self.name!r} enabled={self.enabled}>"


class Event(Base):
    """A persisted analysis result (e.g. detections on a frame)."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    stream_id: Mapped[int | None] = mapped_column(
        ForeignKey("streams.id", ondelete="CASCADE"), nullable=True, index=True
    )
    capability: Mapped[str] = mapped_column(String(64), index=True)
    #: JSON payload of the analyzer result
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, index=True
    )

    stream: Mapped[Stream | None] = relationship(back_populates="events")


class Alert(Base):
    """An alert event broadcast over ``/ws/alerts`` and served by the API.

    ``timestamp`` is the unix epoch seconds at which the alert occurred at
    the source (e.g. when an intruder crossed a virtual fence), while
    ``created_at`` is when the backend received it.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    #: originating inference module, e.g. "detection", "fence", "ocr"
    module: Mapped[str] = mapped_column(String(64), index=True)
    #: "info" | "warning" | "critical" | ...
    severity: Mapped[str] = mapped_column(String(16), index=True)
    message: Mapped[str] = mapped_column(Text)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    timestamp: Mapped[float | None] = mapped_column(Float, nullable=True)
    camera_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    #: base64-encoded frame thumbnail, if any
    thumbnail_base64: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, index=True
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Alert id={self.id} module={self.module!r} severity={self.severity!r}>"


class WatchlistEntry(Base):
    """SQLAlchemy view of the face watchlist.

    This maps the exact ``watchlist`` table written by
    ``inference.face_verification.WatchlistManager`` (raw sqlite3), so both
    the manager (enroll/match via DeepFace) and this ORM model (REST listing)
    stay consistent on the same SQLite file. Do not change the column layout
    without changing ``WatchlistManager`` too.
    """

    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    #: embedding model these rows belong to (e.g. "ArcFace")
    model: Mapped[str] = mapped_column(String(64), index=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    enrolled_at: Mapped[str] = mapped_column(String(32))

    def __repr__(self) -> str:  # pragma: no cover
        return f"<WatchlistEntry id={self.id} name={self.name!r} model={self.model!r}>"
