"""Database engine and session handling."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

DB_URL = os.getenv("CELL_DB_URL", "sqlite:///data/cell.db")

if DB_URL.startswith("sqlite:///"):
    Path(DB_URL.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

# check_same_thread=False so the FastAPI threadpool can share one SQLite file.
_connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
engine = create_engine(DB_URL, connect_args=_connect_args, pool_pre_ping=True)


def init_db() -> None:
    """Create any missing tables. Safe to call on every start."""
    import service.models  # noqa: F401  (registers the tables on SQLModel.metadata)

    SQLModel.metadata.create_all(engine)

    if DB_URL.startswith("sqlite"):
        # WAL lets the dashboard read while the cell is mid-write.
        with engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with Session(engine) as session:
        yield session
