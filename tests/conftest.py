"""Test fixtures.

The database URL is read at import time in service.db, so it has to be redirected
to a scratch file before anything under service/ is imported.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP_DB = Path(tempfile.gettempdir()) / "cell_test.db"
os.environ["CELL_DB_URL"] = f"sqlite:///{_TMP_DB.as_posix()}"


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from service.app import app
    from service.db import engine, init_db

    if _TMP_DB.exists():
        engine.dispose()
        _TMP_DB.unlink()
    init_db()

    with TestClient(app) as c:
        yield c
