"""Watchlist REST endpoints.

Enrollment uses ``WatchlistManager`` from ``inference/face_verification.py``
(RetinaFace detect + ArcFace embedding, persisted to SQLite); listing reads
the same ``watchlist`` table through the SQLAlchemy ``WatchlistEntry`` model.
"""

from __future__ import annotations

import base64
import logging
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ..database import WATCHLIST_DB, WatchlistSessionLocal
from ..models import WatchlistEntry
from ..schemas import (
    WatchlistEnrollRequest,
    WatchlistEnrollResponse,
    WatchlistEntryOut,
)

router = APIRouter(prefix="/watchlist", tags=["watchlist"])
logger = logging.getLogger(__name__)


def _get_manager():
    """Build the Prompt-4 WatchlistManager on the shared SQLite file."""
    from inference.face_verification import WatchlistManager

    return WatchlistManager(db_path=WATCHLIST_DB)


def _decode_image(image_base64: str) -> np.ndarray:
    """Decode a base64 (or data URL) image into a BGR numpy array."""
    if image_base64.startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64") from exc
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400, detail="image_base64 did not decode to a valid image"
        )
    return image


@router.get("", response_model=List[WatchlistEntryOut], summary="List enrolled faces")
def list_watchlist(name: Optional[str] = Query(None, description="filter by name")):
    """Return one entry per enrolled identity (latest row wins for a name)."""
    db = WatchlistSessionLocal()
    try:
        query = db.query(WatchlistEntry).order_by(WatchlistEntry.id)
        if name:
            query = query.filter(WatchlistEntry.name == name)
        seen: set = set()
        entries: List[WatchlistEntry] = []
        for row in query.all():
            if row.name in seen:
                continue
            seen.add(row.name)
            entries.append(row)
        return entries
    finally:
        db.close()


@router.post(
    "/enroll",
    response_model=WatchlistEnrollResponse,
    status_code=201,
    summary="Enroll a face from a name + base64 photo",
)
def enroll_face(payload: WatchlistEnrollRequest) -> WatchlistEnrollResponse:
    """Detect the face in the uploaded photo and store its ArcFace embedding."""
    try:
        image = _decode_image(payload.image_base64)
        manager = _get_manager()
        info = manager.enroll(payload.name, image)
    except HTTPException:
        raise
    except ImportError as exc:
        logger.exception("enroll failed: face models unavailable")
        raise HTTPException(
            status_code=503,
            detail="Face models unavailable - install deepface (requires "
                   "Python 3.10-3.13; not available on 3.14): " + str(exc),
        ) from exc
    except (ValueError, OSError) as exc:
        logger.exception("enroll failed for %r", payload.name)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("enroll unexpected error for %r", payload.name)
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {exc}") from exc

    return WatchlistEnrollResponse(
        name=info["name"],
        bbox=info["bbox"],
        face_confidence=info["face_confidence"],
        faces_in_image=info["faces_in_image"],
        message=f"Enrolled '{info['name']}' into the watchlist",
    )