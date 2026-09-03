"""On-demand analysis endpoints and persisted events."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from inference import list_capabilities

from ..database import get_db
from ..models import Event, Stream
from ..pipeline_manager import pipeline_manager
from ..schemas import EventOut

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _get_stream(db: Session, stream_id: int) -> Stream:
    stream = db.get(Stream, stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    return stream


@router.get("/capabilities", summary="List available inference capabilities")
def capabilities() -> dict:
    return {"capabilities": list_capabilities()}


@router.get(
    "/streams/{stream_id}/results",
    summary="Latest in-memory results for a stream",
)
def latest_results(stream_id: int, db: Session = Depends(get_db)) -> dict:
    _get_stream(db, stream_id)
    snapshot = pipeline_manager.snapshot(stream_id)
    if snapshot is None:
        raise HTTPException(
            status_code=409,
            detail="Stream is not running; start it with POST /streams/{id}/start",
        )
    return snapshot


@router.post(
    "/streams/{stream_id}/run/{capability}",
    summary="Run one capability on the next available frame",
)
def run_capability(stream_id: int, capability: str, db: Session = Depends(get_db)) -> dict:
    if capability not in list_capabilities():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown capability '{capability}'. Available: {list_capabilities()}",
        )
    stream = _get_stream(db, stream_id)
    result = pipeline_manager.run_once(
        stream.id, stream.source_url, stream.capabilities, capability
    )
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="No frame available yet — the source may still be connecting.",
        )
    return result


@router.get("/events", response_model=List[EventOut], summary="Persisted analysis events")
def list_events(
    stream_id: Optional[int] = None,
    capability: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    query = db.query(Event).order_by(Event.id.desc())
    if stream_id is not None:
        query = query.filter(Event.stream_id == stream_id)
    if capability is not None:
        query = query.filter(Event.capability == capability)
    return query.limit(min(limit, 500)).all()
