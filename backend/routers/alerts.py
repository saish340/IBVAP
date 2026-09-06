"""Alert ingest, realtime broadcast, and history endpoints."""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Alert
from ..realtime import broadcaster
from ..schemas import AlertCreate, AlertHistoryOut, AlertOut

router = APIRouter(tags=["alerts"])
logger = logging.getLogger(__name__)


@router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket) -> None:
    """Realtime alert stream.

    Connect from the browser:
        new WebSocket("ws://localhost:8000/ws/alerts")

    The client receives one JSON message per ingested alert. The server
    keeps the socket open and ignores anything the client sends.
    """
    await broadcaster.connect(websocket)
    try:
        # Keep the connection alive; broadcasts are delivered as they arrive.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)
        logger.info("Alerts WebSocket client disconnected")
    except Exception:
        logger.exception("Alerts WebSocket error")
        broadcaster.disconnect(websocket)


@router.post(
    "/events/ingest",
    response_model=AlertOut,
    status_code=201,
    summary="Store an alert event and broadcast it to /ws/alerts clients",
)
async def ingest_event(payload: AlertCreate, db: Session = Depends(get_db)) -> Alert:
    """Persist an alert and immediately push it to every connected WebSocket.

    The event shape is ``{module, severity, message, track_id, timestamp,
    camera_id, thumbnail_base64}``. ``timestamp`` defaults to now (unix
    seconds) when omitted.
    """
    alert = Alert(
        module=payload.module,
        severity=payload.severity,
        message=payload.message,
        track_id=payload.track_id,
        timestamp=payload.timestamp if payload.timestamp is not None else time.time(),
        camera_id=payload.camera_id,
        thumbnail_base64=payload.thumbnail_base64,
        detection_condition=payload.detection_condition,
        detection_reliability_score=payload.detection_reliability_score,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)

    # Fire-and-forget broadcast; never fail the ingest because a client is
    # slow/disconnected.
    try:
        await broadcaster.broadcast(
            AlertOut.model_validate(alert).model_dump(mode="json")
        )
    except Exception:
        logger.exception("alert broadcast failed")

    return alert


@router.get(
    "/events/history",
    response_model=AlertHistoryOut,
    summary="Paginated past alert events",
)
def event_history(
    page: int = Query(1, ge=1, description="1-based page number"),
    page_size: int = Query(20, ge=1, le=200, description="items per page"),
    module: Optional[str] = Query(None, description="filter by source module"),
    severity: Optional[str] = Query(None, description="filter by severity"),
    camera_id: Optional[str] = Query(None, description="filter by camera id"),
    db: Session = Depends(get_db),
) -> dict:
    query = db.query(Alert)
    if module:
        query = query.filter(Alert.module == module)
    if severity:
        query = query.filter(Alert.severity == severity)
    if camera_id:
        query = query.filter(Alert.camera_id == camera_id)

    total = query.count()
    items = (
        query.order_by(Alert.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return {"items": items, "total": total, "page": page, "page_size": page_size}