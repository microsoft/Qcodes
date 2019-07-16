"""
This module provides means of connecting to a QCoDeS database file and
initialising it. Note that connecting/initialisation take into account
database version and possibly perform database upgrades.
"""
import io
import sqlite3
import sys
from os.path import expanduser, normpath
from typing import Union, Tuple, Optional

import numpy as np
from numpy import ndarray

from qcodes.dataset.sqlite.connection import ConnectionPlus
from qcodes.dataset.sqlite.db_upgrades import _latest_available_version, \
    get_user_version, perform_db_upgrade
from qcodes.dataset.sqlite.initial_schema import init_db
import qcodes.config
from qcodes.utils.types import complex_types, complex_type_union


# utility function to allow sqlite/numpy type
def _adapt_array(arr: ndarray) -> sqlite3.Binary:
    """
    See this:
    https://stackoverflow.com/questions/3425320/sqlite3-programmingerror-you-must-not-use-8-bit-bytestrings-unless-you-use-a-te
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def _convert_array(text: bytes) -> ndarray:
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def _convert_complex(text: bytes) -> complex_type_union:
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)[0]


this_session_default_encoding = sys.getdefaultencoding()


def _convert_numeric(value: bytes) -> Union[float, int, str]:
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
    except ValueError as e:
        # If an exception has been raised, we first need to find out
        # if the reason was the conversion to float, and, if so, we are sure
        # that we need to return a string
        if "could not convert string to float" in str(e):
            return str(value, encoding=this_session_default_encoding)
        else:
            # otherwise, the exception is forwarded up the stack
            raise e

    # If that worked, e.g. did not raise an exception, then we check if the
    # outcome is 'nan'
    if np.isnan(numeric):
        return numeric

    # Then we check if the outcome is 'inf', includes +inf and -inf
    if np.isinf(numeric):
        return numeric

    # If it is not 'nan' and not 'inf', then we need to see if the value is
    # really an integer or with floating point digits
    numeric_int = int(numeric)
    if numeric != numeric_int:
        return numeric
    else:
        return numeric_int


def _adapt_float(fl: float) -> Union[float, str]:
    if np.isnan(fl):
        return "nan"
    return float(fl)


def _adapt_complex(value: complex_type_union) -> sqlite3.Binary:
    out = io.BytesIO()
    np.save(out, np.array([value]))
    out.seek(0)
    return sqlite3.Binary(out.read())


def connect(name: str, debug: bool = False,
            version: int = -1) -> ConnectionPlus:
    """
    Connect or create  database. If debug the queries will be echoed back.
    This function takes care of registering the numpy/sqlite type
    converters that we need.

    Args:
        name: name or path to the sqlite file
        debug: whether or not to turn on tracing
        version: which version to create. We count from 0. -1 means 'latest'.
            Should always be left at -1 except when testing.

    Returns:
        conn: connection object to the database (note, it is
            `ConnectionPlus`, not `sqlite3.Connection`

    """
    # register numpy->binary(TEXT) adapter
    # the typing here is ignored due to what we think is a flaw in typeshed
    # see https://github.com/python/typeshed/issues/2429
    sqlite3.register_adapter(np.ndarray, _adapt_array)  # type: ignore
    # register binary(TEXT) -> numpy converter
    # for some reasons mypy complains about this
    sqlite3.register_converter("array", _convert_array)

    sqlite3_conn = sqlite3.connect(name, detect_types=sqlite3.PARSE_DECLTYPES)
    conn = ConnectionPlus(sqlite3_conn)

    latest_supported_version = _latest_available_version()
    db_version = get_user_version(conn)

    if db_version > latest_supported_version:
        raise RuntimeError(f"Database {name} is version {db_version} but this "
                           f"version of QCoDeS supports up to "
                           f"version {latest_supported_version}")

    # sqlite3 options
    conn.row_factory = sqlite3.Row

    # Make sure numpy ints and floats types are inserted properly
    for numpy_int in [
        np.int, np.int8, np.int16, np.int32, np.int64,
        np.uint, np.uint8, np.uint16, np.uint32, np.uint64
    ]:
        sqlite3.register_adapter(numpy_int, int)

    sqlite3.register_converter("numeric", _convert_numeric)

    for numpy_float in [np.float, np.float16, np.float32, np.float64]:
        sqlite3.register_adapter(numpy_float, _adapt_float)

    for complex_type in complex_types:
        sqlite3.register_adapter(complex_type, _adapt_complex)  # type: ignore
    sqlite3.register_converter("complex", _convert_complex)

    if debug:
        conn.set_trace_callback(print)

    init_db(conn)
    perform_db_upgrade(conn, version=version)
    return conn


def get_db_version_and_newest_available_version(path_to_db: str) -> Tuple[int,
                                                                          int]:
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

    return db_version, _latest_available_version()


def get_DB_location() -> str:
    return normpath(expanduser(qcodes.config["core"]["db_location"]))


def get_DB_debug() -> bool:
    return bool(qcodes.config["core"]["db_debug"])


def initialise_database() -> None:
    """
    Initialise a database in the location specified by the config object
    If the database already exists, nothing happens. The database is
    created with or upgraded to the newest version

    Args:
        config: An instance of the config object
    """
    conn = connect(get_DB_location(), get_DB_debug())
    # init is actually idempotent so it's safe to always call!
    init_db(conn)
    conn.close()
    del conn


def initialise_or_create_database_at(db_file_with_abs_path: str) -> None:
    """
    This function sets up QCoDeS to refer to the given database file. If the
    database file does not exist, it will be initiated.

    Args:
        db_file_with_abs_path
            Database file name with absolute path, for example
            ``C:\\mydata\\majorana_experiments.db``
    """
    qcodes.config.core.db_location = db_file_with_abs_path
    initialise_database()


def conn_from_dbpath_or_conn(conn: Optional[ConnectionPlus],
                             path_to_db: Optional[str]) \
        -> ConnectionPlus:
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
