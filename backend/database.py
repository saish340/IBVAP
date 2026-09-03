"""SQLAlchemy engine/session setup.

Defaults to a zero-config SQLite database for local development; set
``DATABASE_URL`` (e.g. ``postgresql+psycopg2://user:pass@host:5432/ibvap``)
to use PostgreSQL — docker-compose does this for the backend container.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ibvap.db")

engine_kwargs: dict = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    # Allow the session to be used from FastAPI's threadpool / worker threads.
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def init_db() -> None:
    """Create tables if they do not exist yet."""
    from . import models  # noqa: F401  (imported so models register on Base)

    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
