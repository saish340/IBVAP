"""IBVAP backend — FastAPI application entry point.

Run locally with:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import SessionLocal, init_db
from .models import Stream
from .pipeline_manager import pipeline_manager
from .routers import alerts, analytics, health, streams, watchlist

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables and resume enabled streams on boot; stop them on shutdown."""
    init_db()
    db = SessionLocal()
    try:
        enabled = db.query(Stream).filter(Stream.enabled.is_(True)).all()
        streams_to_start = [(s.id, s.source_url, s.capabilities or []) for s in enabled]
    finally:
        db.close()
    started = pipeline_manager.autostart_all(streams_to_start)
    logger.info(
        "%s v%s up — %d stream(s) autostarted",
        settings.app_name,
        settings.version,
        started,
    )
    yield
    pipeline_manager.stop_all()
    logger.info("All pipelines stopped")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Real-time video analytics: RTSP ingestion + AI inference over REST/WebSocket.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(streams.router)
app.include_router(analytics.router)
app.include_router(alerts.router)
app.include_router(watchlist.router)


@app.websocket("/ws/streams/{stream_id}")
async def stream_results_ws(websocket: WebSocket, stream_id: int) -> None:
    """Push the latest analysis results for a stream as JSON messages.

    Connect from the browser:
        new WebSocket("ws://localhost:8000/ws/streams/1")
    """
    await websocket.accept()
    logger.info("WebSocket client connected for stream %s", stream_id)
    last_sent: str | None = None
    try:
        while True:
            snapshot = pipeline_manager.snapshot(stream_id)
            if snapshot is not None:
                encoded = json.dumps(snapshot, default=str)
                if encoded != last_sent:
                    await websocket.send_text(encoded)
                    last_sent = encoded
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from stream %s", stream_id)
    except Exception:
        logger.exception("WebSocket error on stream %s", stream_id)
        try:
            await websocket.close()
        except Exception:
            pass
