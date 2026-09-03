"""Health / readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from ..config import settings
from ..database import engine

router = APIRouter(tags=["health"])


@router.get("/", summary="Service banner")
def root() -> dict:
    return {
        "app": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", summary="Health check (also verifies the database)")
def health() -> dict:
    database = "ok"
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        database = "unavailable"
    return {
        "status": "ok" if database == "ok" else "degraded",
        "app": settings.app_name,
        "version": settings.version,
        "database": database,
    }
