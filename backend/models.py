"""SQLAlchemy ORM models."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, DateTime, ForeignKey, String
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
