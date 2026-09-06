"""CRUD + life-cycle endpoints for video streams."""

from __future__ import annotations

from typing import List

import time

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from inference import list_capabilities

from ..database import get_db
from ..models import Stream
from ..pipeline_manager import pipeline_manager
from ..schemas import StreamCreate, StreamOut

router = APIRouter(prefix="/streams", tags=["streams"])


def _get_stream(db: Session, stream_id: int) -> Stream:
    stream = db.get(Stream, stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    return stream


def _to_out(stream: Stream) -> StreamOut:
    return StreamOut(
        id=stream.id,
        name=stream.name,
        source_url=stream.source_url,
        capabilities=stream.capabilities or [],
        enabled=stream.enabled,
        running=pipeline_manager.is_running(stream.id),
    )


@router.get("", response_model=List[StreamOut], summary="List all streams")
def list_streams(db: Session = Depends(get_db)):
    streams = db.query(Stream).order_by(Stream.id).all()
    return [_to_out(stream) for stream in streams]


@router.post(
    "",
    response_model=StreamOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new stream",
)
def create_stream(payload: StreamCreate, db: Session = Depends(get_db)):
    if db.query(Stream).filter(Stream.name == payload.name).first() is not None:
        raise HTTPException(
            status_code=409, detail=f"A stream named '{payload.name}' already exists"
        )
    unknown = [c for c in payload.capabilities if c not in list_capabilities()]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown capabilities: {unknown}. Available: {list_capabilities()}",
        )
    stream = Stream(
        name=payload.name,
        source_url=payload.source_url,
        capabilities=payload.capabilities,
        enabled=payload.auto_start,
    )
    db.add(stream)
    db.commit()
    db.refresh(stream)
    if payload.auto_start:
        pipeline_manager.start(stream.id, stream.source_url, stream.capabilities)
    return _to_out(stream)


@router.get("/{stream_id}", response_model=StreamOut, summary="Get one stream")
def get_stream(stream_id: int, db: Session = Depends(get_db)):
    return _to_out(_get_stream(db, stream_id))


@router.get("/{stream_id}/frame.jpg", summary="Get the latest stream frame")
def latest_frame(stream_id: int):
    pipeline = pipeline_manager.snapshot(stream_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} is not running")
    stream_pipeline = pipeline_manager.get(stream_id)
    jpeg = stream_pipeline.latest_frame_jpeg() if stream_pipeline else None
    if jpeg is None:
        raise HTTPException(status_code=404, detail="No frame available yet")
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/{stream_id}/mjpeg", summary="Continuously stream the latest camera frame")
def mjpeg_stream(stream_id: int):
    """MJPEG is intentionally sourced from the capture buffer, not inference.

    A slow YOLO/face pass therefore only makes overlay data older; it cannot
    pause camera delivery to the browser.
    """
    pipeline = pipeline_manager.get(stream_id)
    if pipeline is None or not pipeline.is_running:
        raise HTTPException(status_code=404, detail="Stream is not running")

    def frames():
        while pipeline.is_running:
            jpeg = pipeline.latest_frame_jpeg()
            if jpeg is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n"
                yield f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                yield jpeg
                yield b"\r\n"
            time.sleep(1 / 12)

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@router.delete("/{stream_id}", summary="Delete a stream and stop its pipeline")
def delete_stream(stream_id: int, db: Session = Depends(get_db)):
    stream = _get_stream(db, stream_id)
    pipeline_manager.stop(stream_id)
    db.delete(stream)
    db.commit()
    return {"status": "deleted", "stream_id": stream_id}


@router.post("/{stream_id}/start", response_model=StreamOut, summary="Start analyzing a stream")
def start_stream(stream_id: int, db: Session = Depends(get_db)):
    stream = _get_stream(db, stream_id)
    pipeline_manager.start(stream.id, stream.source_url, stream.capabilities)
    stream.enabled = True
    db.commit()
    db.refresh(stream)
    return _to_out(stream)


@router.post("/{stream_id}/stop", response_model=StreamOut, summary="Stop analyzing a stream")
def stop_stream(stream_id: int, db: Session = Depends(get_db)):
    stream = _get_stream(db, stream_id)
    pipeline_manager.stop(stream_id)
    stream.enabled = False
    db.commit()
    db.refresh(stream)
    return _to_out(stream)
