"""On-demand analysis endpoints and persisted events."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from inference import list_capabilities

from ..config import settings
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


def _env_gate(capability: str) -> None:
    """Reject env-disabled heavy capabilities with 503 (not silent removal)."""
    gated = (
        (capability in ("face_recognition", "face_verification") and not settings.enable_face)
        or (capability == "pose" and not settings.enable_pose)
        or (capability in ("ocr", "anpr") and not settings.enable_anpr)
    )
    if gated:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Capability '{capability}' is disabled on this host "
                f"(IBVAP_ENABLE_*=0 for the free demo). Enable it and redeploy "
                f"for full functionality — nothing was removed."
            ),
        )


def _get_anpr_engine(stream_id: int):
    """Get the ANPREngine instance for a stream, or None if not available."""
    pipeline = pipeline_manager.get(stream_id)
    if pipeline is None:
        return None
    analyzer = pipeline.analyzers.get("anpr")
    if analyzer is None:
        return None
    # Ensure the model is loaded and return the underlying ANPREngine
    analyzer.ensure_loaded()
    return getattr(analyzer, "_model", None)


@router.get("/capabilities", summary="List available inference capabilities")
def capabilities() -> dict:
    """All registered capabilities, annotated with env-gated availability.

    ``available`` reflects what is *installed + enabled* here; ``disabled_by_env``
    names capabilities switched off via ``IBVAP_ENABLE_*`` (free hosts), and
    ``demo_mode`` mirrors the server mode. Unavailable-but-registered names are
    kept (never silently dropped) so clients can explain *why* — e.g. face
    verification needs DeepFace/TensorFlow, which is too heavy for free CPU.
    """
    all_caps = list_capabilities()
    disabled = [
        name
        for name in all_caps
        if (
            (name in ("face_recognition", "face_verification") and not settings.enable_face)
            or (name == "pose" and not settings.enable_pose)
            or (name in ("ocr", "anpr") and not settings.enable_anpr)
        )
    ]
    return {
        "capabilities": all_caps,
        "available": [c for c in all_caps if c not in disabled],
        "disabled_by_env": disabled,
        "demo_mode": settings.demo_mode,
    }


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
    _env_gate(capability)
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


@router.post(
    "/streams/{stream_id}/plates/{plate_text}/suppress",
    summary="Suppress a number plate so its box is no longer drawn",
)
def suppress_plate(stream_id: int, plate_text: str, db: Session = Depends(get_db)) -> dict:
    """Suppress a plate so its bounding box is no longer drawn on frames."""
    _get_stream(db, stream_id)
    engine = _get_anpr_engine(stream_id)
    if engine is None:
        raise HTTPException(
            status_code=409,
            detail="ANPR is not enabled for this stream",
        )
    engine.suppress_plate(plate_text)
    return {"status": "suppressed", "plate": plate_text.upper(), "stream_id": stream_id}


@router.delete(
    "/streams/{stream_id}/plates/{plate_text}/suppress",
    summary="Unsuppress a number plate so its box is drawn again",
)
def unsuppress_plate(stream_id: int, plate_text: str, db: Session = Depends(get_db)) -> dict:
    """Unsuppress a plate so its bounding box is drawn again on frames."""
    _get_stream(db, stream_id)
    engine = _get_anpr_engine(stream_id)
    if engine is None:
        raise HTTPException(
            status_code=409,
            detail="ANPR is not enabled for this stream",
        )
    engine.unsuppress_plate(plate_text)
    return {"status": "unsuppressed", "plate": plate_text.upper(), "stream_id": stream_id}


@router.get(
    "/streams/{stream_id}/plates/suppressed",
    summary="List all suppressed plates for a stream",
)
def list_suppressed_plates(stream_id: int, db: Session = Depends(get_db)) -> dict:
    """List all currently suppressed plates for a stream."""
    _get_stream(db, stream_id)
    engine = _get_anpr_engine(stream_id)
    if engine is None:
        raise HTTPException(
            status_code=409,
            detail="ANPR is not enabled for this stream",
        )
    return {"suppressed_plates": list(engine._suppressed_plates), "stream_id": stream_id}


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
