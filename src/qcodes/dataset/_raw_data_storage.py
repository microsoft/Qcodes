"""
Module for managing per-dataset raw data SQLite files.

When the ``dataset.raw_data_to_separate_db`` config option is enabled,
measurement data (results tables) are written to individual SQLite files
- one per dataset - instead of the main QCoDeS database file.  All metadata
(runs, experiments, layouts, dependencies) remains in the main database.

The per-dataset files are stored in the folder given by
``dataset.raw_data_path`` and are named ``<guid>.db``.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import qcodes
from qcodes.dataset.export_config import _expand_export_path
from qcodes.dataset.sqlite.connection import AtomicConnection
from qcodes.dataset.sqlite.database import (
    _adapt_array,
    _adapt_complex,
    _adapt_float,
    _convert_array,
    _convert_complex,
    _convert_numeric,
)
from qcodes.utils.types import complex_types, numpy_floats, numpy_ints

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qcodes.parameters import ParamSpecBase

log = logging.getLogger(__name__)

_RAW_DATA_CONFIG_SECTION = "dataset"
_RAW_DATA_ENABLED_KEY = "raw_data_to_separate_db"
_RAW_DATA_PATH_KEY = "raw_data_path"


def is_raw_data_storage_enabled() -> bool:
    """Return True if per-dataset raw data storage is enabled in config."""
    return bool(
        qcodes.config[_RAW_DATA_CONFIG_SECTION].get(_RAW_DATA_ENABLED_KEY, False)
    )


def get_raw_data_folder() -> Path:
    """Return the resolved folder path for raw data SQLite files.

    The path template from config is expanded the same way as the
    export path (``{db_location}`` is replaced with a folder derived
    from the main database path).
    """
    raw_path_template: str = qcodes.config[_RAW_DATA_CONFIG_SECTION].get(
        _RAW_DATA_PATH_KEY, "{db_location}"
    )
    return Path(_expand_export_path(raw_path_template)).expanduser().absolute()


def get_raw_data_db_path(guid: str, folder: Path | None = None) -> Path:
    """Return the full path for a dataset's raw data SQLite file.

    Args:
        guid: The GUID of the dataset.
        folder: Override folder.  If *None*, uses :func:`get_raw_data_folder`.

    """
    if folder is None:
        folder = get_raw_data_folder()
    return folder / f"{guid}.db"


def connect_to_raw_data_db(
    path: str | Path,
    *,
    read_only: bool = False,
) -> AtomicConnection:
    """Open (or create) a lightweight SQLite connection for raw data.

    Unlike the main QCoDeS :func:`~qcodes.dataset.sqlite.database.connect`,
    this does **not** create the full metadata schema (experiments, runs, ...).
    It only registers the numpy/sqlite type adapters that QCoDeS needs to
    round-trip array and numeric data.

    Args:
        path: Path to the raw-data SQLite file.
        read_only: Open the database in read-only mode.

    Returns:
        An :class:`AtomicConnection` to the raw-data database.

    """
    # Register adapters/converters (idempotent calls)
    sqlite3.register_adapter(np.ndarray, _adapt_array)
    sqlite3.register_converter("array", _convert_array)
    for numpy_int in numpy_ints:
        sqlite3.register_adapter(numpy_int, int)
    sqlite3.register_converter("numeric", _convert_numeric)
    for numpy_float in (float, *numpy_floats):
        sqlite3.register_adapter(numpy_float, _adapt_float)
    for complex_type in complex_types:
        sqlite3.register_adapter(complex_type, _adapt_complex)  # type: ignore[arg-type]
    sqlite3.register_converter("complex", _convert_complex)

    uri = f"file:{path!s}"
    if read_only:
        uri += "?mode=ro"

    conn = sqlite3.connect(
        uri,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=True,
        uri=True,
        factory=AtomicConnection,
    )
    return conn


def create_raw_data_db(
    path: str | Path,
    table_name: str,
    paramspecs: Sequence[ParamSpecBase],
) -> AtomicConnection:
    """Create a per-dataset raw-data SQLite file with a results table.

    The file is created if it does not exist.  The parent directory is
    created if needed.

    Args:
        path: Full path for the new SQLite file.
        table_name: Name of the results table to create (matches the name
            in the main database).
        paramspecs: Parameter specifications describing the columns.

    Returns:
        An :class:`AtomicConnection` to the newly created database.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect_to_raw_data_db(path)

    if paramspecs:
        columns = ",".join(f'"{p.name}" {p.type}' for p in paramspecs)
        sql = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id INTEGER PRIMARY KEY,
            {columns}
        );
        """
    else:
        sql = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id INTEGER PRIMARY KEY
        );
        """

    conn.execute(sql)
    conn.commit()

    log.info(
        "Created raw data database at %s with table %s",
        path,
        table_name,
    )
    return conn
