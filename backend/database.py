"""SQLAlchemy engine/session setup.

Defaults to a zero-config SQLite database for local development; set
``DATABASE_URL`` (e.g. ``postgresql+psycopg2://user:pass@host:5432/ibvap``)
to use PostgreSQL — docker-compose does this for the backend container.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ibvap.db")

engine_kwargs: dict = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    # Allow the session to be used from FastAPI's threadpool / worker threads.
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# --- Face watchlist database -------------------------------------------------
# The watchlist always lives in its own SQLite file (sqlite is how the
# WatchlistManager persists). The ORM model `WatchlistEntry` maps the same
# table so REST listing and DeepFace matching share one store.
WATCHLIST_DB = settings.watchlist_db_path
watchlist_engine = create_engine(
    f"sqlite:///{WATCHLIST_DB}", connect_args={"check_same_thread": False}
)
WatchlistSessionLocal = sessionmaker(bind=watchlist_engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def init_db() -> None:
    """Create tables if they do not exist yet.

    The main tables (streams, events, alerts) go on the primary engine; the
    ``watchlist`` table goes on the dedicated watchlist SQLite file so it
    stays in sync with ``WatchlistManager``.
    """
    from . import models  # noqa: F401  (imported so models register on Base)

    main_tables = [t for t in Base.metadata.sorted_tables if t.name != "watchlist"]
    Base.metadata.create_all(bind=engine, tables=main_tables)
    Base.metadata.tables["watchlist"].create(bind=watchlist_engine, checkfirst=True)


def get_db():
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
