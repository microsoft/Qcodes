"""
This module provides means of connecting to a QCoDeS database file and
initialising it. Note that connecting/initialisation take into account
database version and possibly perform database upgrades.
"""
from __future__ import annotations

import io
import math
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from os.path import expanduser, normpath
from pathlib import Path
from typing import Literal

import numpy as np

import qcodes
from qcodes.dataset.experiment_settings import reset_default_experiment_id
from qcodes.dataset.sqlite.connection import ConnectionPlus
from qcodes.dataset.sqlite.db_upgrades import (
    _latest_available_version,
    perform_db_upgrade,
)
from qcodes.dataset.sqlite.db_upgrades.version import get_user_version
from qcodes.dataset.sqlite.initial_schema import init_db
from qcodes.utils.types import complex_types, numpy_floats, numpy_ints

JournalMode = Literal["DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"]


# utility function to allow sqlite/numpy type
def _adapt_array(arr: np.ndarray) -> sqlite3.Binary:
    """
    See this:
    https://stackoverflow.com/questions/3425320/sqlite3-programmingerror-you-must-not-use-8-bit-bytestrings-unless-you-use-a-te
    """
    out = io.BytesIO()
    # Directly use np.lib.format.write_array instead of np.save, force version to be
    # 3.0 (when reading, version 1.0 and 2.0 can result in a slow clean up step to
    # ensure backward compatibility with python 2) and disable pickle (slow and
    # insecure)
    np.lib.format.write_array(out, arr, version=(3, 0), allow_pickle=False)
    out.seek(0)
    return sqlite3.Binary(out.read())


def _convert_array(text: bytes) -> np.ndarray:
    # Using np.lib.format.read_array (counterpart of np.lib.format.write_array)
    # npy format version 3.0 is 3 times faster than previous verions (no clean up step
    # for python 2 backward compatibility)
    return np.lib.format.read_array(io.BytesIO(text), allow_pickle=False)


def _convert_complex(text: bytes) -> np.complexfloating:
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)[0]


this_session_default_encoding = sys.getdefaultencoding()


def _convert_numeric(value: bytes) -> float | int | str:
    """
    This is a converter for sqlite3 'numeric' type class.

    This converter is capable of deducting whether a number is a float or an
    int.

    Note sqlite3 allows to save data to columns even if their type is not
    compatible with the table type class (for example, it is possible to save
    integers into 'text' columns). Due to this fact, and for the reasons of
    flexibility, the numeric converter is also made capable of handling
    strings. An obvious exception to this is 'nan' (case insensitive) which
    gets converted to `np.nan`. Another exception to this is 'inf', which
    gets converted to 'np.inf'.
    """
    try:
        # First, try to convert bytes to float
        numeric = float(value)
    except ValueError:
        # Let string casting fail if bytes encoding is invalid
        return str(value, encoding=this_session_default_encoding)

    # If that worked, e.g. did not raise an exception, then we check if the outcome is
    # either an infinity or a NaN
    # For a single value, math.isfinite is 10 times faster than np.isinfinite (or
    # combining np.isnan and np.isinf)
    if not math.isfinite(numeric):
        return numeric

    # If it is not 'nan' and not 'inf', then we need to see if the value is really an
    # integer or with floating point digits
    numeric_int = int(numeric)
    if numeric != numeric_int:
        return numeric
    else:
        return numeric_int

def _adapt_float(fl: float) -> float | str:
    # For a single value, math.isnan is 10 times faster than np.isnan
    # Overall, saving floats with numeric format is 2 times faster with math.isnan
    if math.isnan(fl):
        return "nan"
    return float(fl)


def _adapt_complex(value: complex | np.complexfloating) -> sqlite3.Binary:
    out = io.BytesIO()
    np.save(out, np.array([value]))
    out.seek(0)
    return sqlite3.Binary(out.read())


def connect(name: str | Path, debug: bool = False, version: int = -1) -> ConnectionPlus:
    """
    Connect or create  database. If debug the queries will be echoed back.
    This function takes care of registering the numpy/sqlite type
    converters that we need.

    Args:
        name: name or path to the sqlite file
        debug: should tracing be turned on.
        version: which version to create. We count from 0. -1 means 'latest'.
            Should always be left at -1 except when testing.

    Returns:
        connection object to the database (note, it is
        :class:`ConnectionPlus`, not :class:`sqlite3.Connection`)

    """
    # register numpy->binary(TEXT) adapter
    sqlite3.register_adapter(np.ndarray, _adapt_array)
    # register binary(TEXT) -> numpy converter
    sqlite3.register_converter("array", _convert_array)

    sqlite3_conn = sqlite3.connect(name, detect_types=sqlite3.PARSE_DECLTYPES,
                                   check_same_thread=True)
    conn = ConnectionPlus(sqlite3_conn)

    latest_supported_version = _latest_available_version()
    db_version = get_user_version(conn)

    if db_version > latest_supported_version:
        raise RuntimeError(f"Database {name} is version {db_version} but this "
                           f"version of QCoDeS supports up to "
                           f"version {latest_supported_version}")

    # Make sure numpy ints and floats types are inserted properly
    for numpy_int in numpy_ints:
        sqlite3.register_adapter(numpy_int, int)

    sqlite3.register_converter("numeric", _convert_numeric)

    for numpy_float in (float,) + numpy_floats:
        sqlite3.register_adapter(numpy_float, _adapt_float)

    for complex_type in complex_types:
        sqlite3.register_adapter(complex_type, _adapt_complex)
    sqlite3.register_converter("complex", _convert_complex)

    if debug:
        conn.set_trace_callback(print)

    init_db(conn)
    perform_db_upgrade(conn, version=version)
    return conn


def get_db_version_and_newest_available_version(
    path_to_db: str | Path,
) -> tuple[int, int]:
    """
    Connect to a DB without performing any upgrades and get the version of
    that database file along with the newest available version (the one that
    a normal "connect" will automatically upgrade to)

    Args:
        path_to_db: the absolute path to the DB file

    Returns:
        A tuple of (db_version, latest_available_version)
    """
    conn = connect(path_to_db, version=0)
    db_version = get_user_version(conn)
    conn.close()

    return db_version, _latest_available_version()


def get_DB_location() -> str:
    return normpath(expanduser(qcodes.config["core"]["db_location"]))


def get_DB_debug() -> bool:
    return bool(qcodes.config["core"]["db_debug"])


def initialise_database(journal_mode: JournalMode | None = "WAL") -> None:
    """
    Initialise a database in the location specified by the config object
    and set ``atomic commit and rollback mode`` of the db. The db is created
    with the latest supported version. If the database already exists the
    ``atomic commit and rollback mode`` is set and the database is upgraded
    to the latest version.

    Args:
        journal_mode: Which `journal_mode` should be used for atomic commit and rollback.
            Options are DELETE, TRUNCATE, PERSIST, MEMORY, WAL and OFF. If set to None
            no changes are made.
    """
    # calling connect performs all the needed actions to create and upgrade
    # the db to the latest version.
    conn = connect(get_DB_location(), get_DB_debug())
    reset_default_experiment_id(conn)
    if journal_mode is not None:
        set_journal_mode(conn, journal_mode)
    conn.close()
    del conn


def set_journal_mode(conn: ConnectionPlus, journal_mode: JournalMode) -> None:
    """
    Set the ``atomic commit and rollback mode`` of the sqlite database.
    See https://www.sqlite.org/pragma.html#pragma_journal_mode for details.

    Args:
        conn: Connection to the database.
        journal_mode: Which `journal_mode` should be used for atomic commit and rollback.
            Options are DELETE, TRUNCATE, PERSIST, MEMORY, WAL and OFF.
    """
    valid_journal_modes = ["DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"]
    if journal_mode not in valid_journal_modes:
        raise RuntimeError(f"Invalid journal_mode {journal_mode} "
                           f"Valid modes are {valid_journal_modes}")
    query = f"PRAGMA journal_mode={journal_mode};"
    cursor = conn.cursor()
    cursor.execute(query)


def initialise_or_create_database_at(
    db_file_with_abs_path: str | Path, journal_mode: JournalMode | None = "WAL"
) -> None:
    """
    This function sets up QCoDeS to refer to the given database file. If the
    database file does not exist, it will be initiated.

    Args:
        db_file_with_abs_path
            Database file name with absolute path, for example
            ``C:\\mydata\\majorana_experiments.db``
        journal_mode: Which `journal_mode` should be used for atomic commit and rollback.
            Options are DELETE, TRUNCATE, PERSIST, MEMORY, WAL and OFF. If set to None
            no changes are made.
    """
    qcodes.config.core.db_location = str(db_file_with_abs_path)
    initialise_database(journal_mode)


@contextmanager
def initialised_database_at(db_file_with_abs_path: str | Path) -> Iterator[None]:
    """
    Initializes or creates a database and restores the 'db_location' afterwards.

    Args:
        db_file_with_abs_path
            Database file name with absolute path, for example
            ``C:\\mydata\\majorana_experiments.db``
    """
    db_location = qcodes.config["core"]["db_location"]
    try:
        initialise_or_create_database_at(db_file_with_abs_path)
        yield
    finally:
        qcodes.config["core"]["db_location"] = db_location


def conn_from_dbpath_or_conn(
    conn: ConnectionPlus | None, path_to_db: str | Path | None
) -> ConnectionPlus:
    """
    A small helper function to abstract the logic needed for functions
    that take either a `ConnectionPlus` or the path to a db file.
    If neither is given this will fall back to the default db location.
    It is an error to supply both.

    Args:
        conn: A ConnectionPlus object pointing to a sqlite database
        path_to_db: The path to a db file.

    Returns:
        A `ConnectionPlus` object
    """

    if path_to_db is not None and conn is not None:
        raise ValueError('Received BOTH conn and path_to_db. Please '
                         'provide only one or the other.')
    if conn is None and path_to_db is None:
        path_to_db = get_DB_location()

    if conn is None and path_to_db is not None:
        conn = connect(path_to_db, get_DB_debug())
    elif conn is not None:
        conn = conn
    else:
        # this should be impossible but left here to keep mypy happy.
        raise RuntimeError("Could not obtain a connection from"
                           "supplied information.")
    return conn
