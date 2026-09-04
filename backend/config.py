"""Application configuration, read from environment variables."""

from __future__ import annotations

import os


def _env_list(key: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(key, default).split(",") if item.strip()]


class Settings:
    """Runtime settings for the IBVAP backend (12-factor style env vars)."""

    def __init__(self) -> None:
        self.app_name: str = "IBVAP Backend"
        self.version: str = "0.1.0"

        # Database. Defaults to a zero-config SQLite file for local dev;
        # docker-compose sets DATABASE_URL to the postgres container.
        self.database_url: str = os.getenv("DATABASE_URL", "sqlite:///./ibvap.db")

        # CORS: the Vite dev server runs on 5173 by default.
        self.cors_origins: list[str] = _env_list(
            "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
        )

        # Ingestion
        self.rtsp_reconnect_delay: float = float(os.getenv("RTSP_RECONNECT_DELAY", "2"))

        # Inference
        self.yolo_model: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
        # Run analyzers on every Nth frame (1 = every frame).
        self.process_every_n_frames: int = max(1, int(os.getenv("PROCESS_EVERY_N_FRAMES", "5")))
        # How often (per capability) the latest result is persisted as an Event.
        self.persist_interval_seconds: float = float(os.getenv("PERSIST_INTERVAL_SECONDS", "10"))

        # Face watchlist SQLite file (shared by WatchlistManager + /watchlist API).
        self.watchlist_db_path: str = os.getenv("WATCHLIST_DB", "watchlist.db")


settings = Settings()
