"""
This module provides means of connecting to a QCoDeS database file and
initialising it. Note that connecting/initialisation take into account
database version and possibly perform database upgrades.
"""

from __future__ import annotations

import functools
import io
import math
import sqlite3
import sys
from contextlib import contextmanager
from os.path import expanduser, normpath
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

import qcodes
from qcodes.dataset.experiment_settings import reset_default_experiment_id
from qcodes.dataset.sqlite.connection import AtomicConnection
from qcodes.dataset.sqlite.db_upgrades import (
    _latest_available_version,
    perform_db_upgrade,
)
from qcodes.dataset.sqlite.db_upgrades.version import get_user_version
from qcodes.dataset.sqlite.initial_schema import init_db
from qcodes.utils.types import complex_types, numpy_floats, numpy_ints

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

JournalMode = Literal["DELETE", "TRUNCATE", "PERSIST", "MEMORY", "WAL", "OFF"]


# utility function to allow sqlite/numpy type
def _adapt_array(arr: npt.NDArray) -> sqlite3.Binary:
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


def _convert_array(text: bytes) -> npt.NDArray:
    """
    Convert the ``.npy`` serialization of an array (as written by
    :func:`_adapt_array`) back into an array.

    This is equivalent to :func:`numpy.lib.format.read_array` but caches the
    parsing of the header, which otherwise dominates the time it takes to
    load small and medium sized arrays. All rows of a given array parameter
    typically share the same header. Input that cannot be handled by the fast
    path is passed on to :func:`numpy.lib.format.read_array`.
    """
    header_info = _npy_header_info(text)
    if header_info is None:
        return np.lib.format.read_array(io.BytesIO(text), allow_pickle=False)
    offset, shape, fortran_order, dtype = header_info
    count = math.prod(shape)
    if len(text) - offset < count * dtype.itemsize:
        return np.lib.format.read_array(io.BytesIO(text), allow_pickle=False)
    # copy since frombuffer returns a read only view of the bytes
    array = np.frombuffer(text, dtype=dtype, count=count, offset=offset).copy()
    if fortran_order:
        return array.reshape(shape[::-1]).transpose()
    return array.reshape(shape)


_NPY_MAGIC_PREFIX = b"\x93NUMPY"


def _npy_header_info(
    text: bytes,
) -> tuple[int, tuple[int, ...], bool, np.dtype[Any]] | None:
    """
    Return the offset of the data, shape, fortran order and dtype of an
    ``.npy`` serialized array or None if the array should be read with
    :func:`numpy.lib.format.read_array`.
    """
    if len(text) < 12 or not text.startswith(_NPY_MAGIC_PREFIX):
        return None
    major_version = text[6]
    if major_version == 1:
        offset = 10 + int.from_bytes(text[8:10], "little")
    elif major_version in (2, 3):
        offset = 12 + int.from_bytes(text[8:12], "little")
    else:
        return None
    header_info = _parse_npy_header(text[:offset])
    if header_info is None:
        return None
    return (offset, *header_info)


@functools.lru_cache(maxsize=128)
def _parse_npy_header(
    header: bytes,
) -> tuple[tuple[int, ...], bool, np.dtype[Any]] | None:
    # numpy's public header readers decode the header as latin1 while
    # version 3.0 headers are utf8 encoded. These are only identical for
    # ascii, which covers all headers written by QCoDeS.
    header_text_start = 10 if header[6] == 1 else 12
    if not header[header_text_start:].isascii():
        return None
    fp = io.BytesIO(header)
    try:
        major_version, _ = np.lib.format.read_magic(fp)
        if major_version == 1:
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fp)
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fp)
    except ValueError:
        return None
    if dtype.hasobject or dtype.itemsize == 0:
        return None
    return shape, fortran_order, dtype


def _convert_complex(text: bytes) -> np.complexfloating:
    return _convert_array(text)[0]


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

    # is_integer is False for NaN and infinity so these are returned as floats
    if numeric.is_integer():
        return int(numeric)
    return numeric


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


def _register_numpy_sqlite_adapters_and_converters() -> None:
    """
    Register the numpy/sqlite type adapters and converters that QCoDeS needs
    to round-trip array, numeric and complex data. The registrations are
    global to the ``sqlite3`` module and idempotent, so calling this multiple
    times is safe.
    """
    # register numpy->binary(TEXT) adapter
    sqlite3.register_adapter(np.ndarray, _adapt_array)
    # register binary(TEXT) -> numpy converter
    sqlite3.register_converter("array", _convert_array)

    # Make sure numpy ints and floats types are inserted properly
    for numpy_int in numpy_ints:
        sqlite3.register_adapter(numpy_int, int)

    sqlite3.register_converter("numeric", _convert_numeric)

    for numpy_float in (float, *numpy_floats):
        sqlite3.register_adapter(numpy_float, _adapt_float)

    for complex_type in complex_types:
        # https://github.com/python/typeshed/issues/2429
        sqlite3.register_adapter(complex_type, _adapt_complex)  # type: ignore[arg-type]
    sqlite3.register_converter("complex", _convert_complex)


def _connect_to_sqlite_file(
    name: str | Path, read_only: bool = False
) -> AtomicConnection:
    """
    Open a bare :class:`AtomicConnection` to an sqlite file with the settings
    that QCoDeS relies on (URI mode, ``PARSE_DECLTYPES``) and the numpy type
    adapters/converters registered.

    Unlike :func:`connect`, this does **not** create or upgrade any QCoDeS
    schema; it is the low-level building block shared by :func:`connect` and
    the per-dataset raw-data connection helper.

    Args:
        name: name or path to the sqlite file
        read_only: Should the database be opened in read only mode.

    Returns:
        connection object to the database (note, it is
        :class:`AtomicConnection`, which is a subclass of :class:`sqlite3.Connection`)

    """
    _register_numpy_sqlite_adapters_and_converters()

    path = f"file:{name!s}"

    if read_only:
        path = path + "?mode=ro"

    return sqlite3.connect(
        path,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=True,
        uri=True,
        factory=AtomicConnection,
    )


def connect(
    name: str | Path, debug: bool = False, version: int = -1, read_only: bool = False
) -> AtomicConnection:
    """
    Connect or create  database. If debug the queries will be echoed back.
    This function takes care of registering the numpy/sqlite type
    converters that we need.

    Args:
        name: name or path to the sqlite file
        debug: should tracing be turned on.
        version: which version to create. We count from 0. -1 means 'latest'.
            Should always be left at -1 except when testing.
        read_only: Should the database be opened in read only mode.

    Returns:
        connection object to the database (note, it is
        :class:`AtomicConnection`, which is a subclass of :class:`sqlite3.Connection`)

    """
    conn = _connect_to_sqlite_file(name, read_only=read_only)

    latest_supported_version = _latest_available_version()
    db_version = get_user_version(conn)

    if db_version > latest_supported_version:
        raise RuntimeError(
            f"Database {name} is version {db_version} but this "
            f"version of QCoDeS supports up to "
            f"version {latest_supported_version}"
        )

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


def initialise_database(
    journal_mode: JournalMode | None = "WAL",
    db_path: str | Path | None = None,
) -> None:
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
        db_path: Path to database file. If None, the path specified in the config object is used.

    """
    # calling connect performs all the needed actions to create and upgrade
    # the db to the latest version.
    if db_path is None:
        db_path = get_DB_location()
    conn = connect(db_path, get_DB_debug())
    reset_default_experiment_id(conn)
    if journal_mode is not None:
        set_journal_mode(conn, journal_mode)
    conn.close()
    del conn


def initialise_or_create_database_at(
    db_file_with_abs_path: str | Path, journal_mode: JournalMode | None = "WAL"
) -> None:
    r"""
    Initialises or creates a database at the specified location and configures QCoDeS to use
    this as the default database for the duration of the session.


    Args:
        db_file_with_abs_path: Database file name with absolute path, for example
            ``C:\mydata\majorana_experiments.db``
        journal_mode: Which `journal_mode` should be used for atomic commit and rollback.
            Options are DELETE, TRUNCATE, PERSIST, MEMORY, WAL and OFF. If set to None
            no changes are made.

    """
    qcodes.config.core.db_location = str(db_file_with_abs_path)
    initialise_database(journal_mode)


@contextmanager
def initialised_database_at(
    db_file_with_abs_path: str | Path, *, journal_mode: JournalMode | None = "WAL"
) -> Generator[None, None, None]:
    r"""
    Initialises or creates a database at the specified location, configures QCoDeS to use this as the
    default database for the duration of the context, and restores the 'db_location' afterwards.

    Args:
        db_file_with_abs_path: Database file name with absolute path, for example
            ``C:\mydata\majorana_experiments.db``
        journal_mode: Which `journal_mode` should be used for atomic commit and rollback.
            Options are DELETE, TRUNCATE, PERSIST, MEMORY, WAL and OFF. If set to None
            no changes are made.

    """
    db_location = qcodes.config["core"]["db_location"]
    try:
        initialise_or_create_database_at(
            db_file_with_abs_path, journal_mode=journal_mode
        )
        yield
    finally:
        qcodes.config["core"]["db_location"] = db_location


def set_journal_mode(conn: AtomicConnection, journal_mode: JournalMode) -> None:
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
        raise RuntimeError(
            f"Invalid journal_mode {journal_mode} Valid modes are {valid_journal_modes}"
        )
    query = f"PRAGMA journal_mode={journal_mode};"
    cursor = conn.cursor()
    cursor.execute(query)


def conn_from_dbpath_or_conn(
    conn: AtomicConnection | None,
    path_to_db: str | Path | None,
    read_only: bool = False,
) -> AtomicConnection:
    """
    A small helper function to abstract the logic needed for functions
    that take either an `AtomicConnection` or the path to a db file.
    If neither is given this will fall back to the default db location.
    It is an error to supply both.

    Args:
        conn: A AtomicConnection object pointing to a sqlite database
        path_to_db: The path to a db file.
        read_only: whether to open the connection in read-only mode.
            Only takes effect if `conn` is not given.

    Returns:
        A `AtomicConnection` object

    """

    if path_to_db is not None and conn is not None:
        raise ValueError(
            "Received BOTH conn and path_to_db. Please provide only one or the other."
        )
    if conn is None and path_to_db is None:
        path_to_db = get_DB_location()

    if conn is None and path_to_db is not None:
        conn = connect(path_to_db, get_DB_debug(), read_only=read_only)
    elif conn is not None:
        pass
    else:
        # this should be impossible but left here to keep mypy happy.
        raise RuntimeError("Could not obtain a connection from supplied information.")
    return conn
