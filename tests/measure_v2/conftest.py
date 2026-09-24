"""Shared fixtures for the measure_v2 test suite."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import pytest

import qcodes as qc
from qcodes.dataset.sqlite.database import initialise_or_create_database_at

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(scope="function")
def empty_db(tmp_path: Path) -> Generator[None, None, None]:
    """Configure qcodes to use an empty SQLite database in tmp_path.

    Initializes the database file so the SqliteSink can immediately open
    runs against it.
    """
    original = qc.config["core"]["db_location"]
    db_path = tmp_path / "measure_v2_test.db"
    try:
        qc.config["core"]["db_location"] = str(db_path)
        initialise_or_create_database_at(str(db_path))
        yield
    finally:
        qc.config["core"]["db_location"] = original
        # Close any leftover SQLite connections before tmp_path teardown.
        gc.collect()
