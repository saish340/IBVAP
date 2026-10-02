"""IBVAP backend — FastAPI application entry point.

Run locally with:
    uvicorn backend.main:app --reload --port 8000

Cloud (Hugging Face Spaces / Railway / Render) honours ``$PORT``:
    uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
(see ``scripts/start_server.sh`` — never bind 127.0.0.1 in the cloud).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

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


def _seed_demo_stream() -> int:
    """Register the bundled sample clip as a demo stream (demo mode only).

    Returns 1 when a stream was seeded, else 0. Never raises — a missing
    sample file must not crash the free demo boot.
    """
    name = settings.demo_stream_name
    video = settings.demo_video_path
    if not video:
        return 0
    try:
        if not Path(video).exists():
            logger.warning("Demo seed skipped: sample file %r not found", video)
            return 0
        db = SessionLocal()
        try:
            exists = db.query(Stream).filter(Stream.name == name).first()
            if exists is not None:
                return 0
            stream = Stream(
                name=name,
                source_url=video,
                capabilities=list(settings.demo_capabilities),
                enabled=True,
            )
            db.add(stream)
            db.commit()
            db.refresh(stream)
            logger.info("Seeded demo stream %r -> %r", name, video)
            return 1
        finally:
            db.close()
    except Exception:
        logger.exception("Demo seed failed (non-fatal)")
        return 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables, optionally seed/autostart streams, stop on shutdown.

    Behaviour by env:
    - ``IBVAP_DEMO_MODE=1`` (+ bundled sample present): seed one
      video-file demo stream so the free demo works with zero setup.
      Webcam (``"0"``) / localhost RTSP sources are NEVER auto-seeded —
      they do not exist on cloud hosts.
    - ``IBVAP_AUTOSTART_STREAMS=0``: skip autostart entirely (cheap boot on
      free hosts that sleep/restart often). Default is to autostart, so
      VPS/Render behaviour is unchanged.
    """
    init_db()
    autostart: list[tuple] = []
    if settings.demo_mode:
        _seed_demo_stream()
    if settings.autostart_streams:
        db = SessionLocal()
        try:
            enabled = db.query(Stream).filter(Stream.enabled.is_(True)).all()
            autostart = [
                (s.id, s.source_url, s.capabilities or []) for s in enabled
            ]
        finally:
            db.close()
        if settings.demo_mode:
            # Cloud demo safety: never autostart hardware/localhost sources —
            # they would spin in reconnect loops on a host with no camera.
            skipped: list[tuple] = []
            kept: list[tuple] = []
            for item in autostart:
                source = str(item[1] or "").strip()
                lowered = source.lower()
                is_hardware = (
                    source.isdigit()
                    or lowered.startswith(("rtsp://", "rtsps://"))
                    or "localhost" in lowered
                    or lowered.startswith("127.")
                )
                (skipped if is_hardware else kept).append(item)
            for stream_id, source_url, _caps in skipped:
                logger.warning(
                    "Demo mode: not autostarting hardware/localhost stream "
                    "%s (%r) — register a video-file stream instead",
                    stream_id, source_url,
                )
            autostart = kept
    started = pipeline_manager.autostart_all(autostart) if autostart else 0
    logger.info(
        "%s v%s up — demo_mode=%s autostart=%s — %d stream(s) autostarted "
        "(port %s)",
        settings.app_name,
        settings.version,
        settings.demo_mode,
        settings.autostart_streams,
        started,
        os.getenv("PORT", os.getenv("BACKEND_PORT", "8000")),
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

    Same-origin (nginx): ``new WebSocket("ws(s)://<host>/ws/streams/1")``.
    Split deploy (Vercel + HF/Render): build the URL from ``VITE_WS_URL``,
    e.g. ``new WebSocket("wss://<backend>/ws/streams/1")`` — never hardcode
    localhost in the frontend (see ``frontend/src/App.jsx``).
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
