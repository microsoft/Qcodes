import sys
from contextlib import contextmanager
import logging
import sqlite3
import time
import io
import warnings
from typing import (Any, List, Optional, Tuple, Union, Dict, cast, Callable,
                    Sequence, DefaultDict)
import itertools
from functools import wraps
from collections import defaultdict

from tqdm import tqdm
from numbers import Number
from numpy import ndarray
import numpy as np
from distutils.version import LooseVersion
import wrapt

import qcodes as qc
import unicodedata
from qcodes.dataset.dependencies import InterDependencies
from qcodes.dataset.descriptions import RunDescriber
from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.guids import generate_guid, parse_guid

log = logging.getLogger(__name__)

# represent the type of  data we can/want map to sqlite column
VALUE = Union[str, Number, List, ndarray, bool]
VALUES = List[VALUE]


# Functions decorated as 'upgrader' are inserted into this dict
# The newest database version is thus determined by the number of upgrades
# in this module
# The key is the TARGET VERSION of the upgrade, i.e. the first key is 1
_UPGRADE_ACTIONS: Dict[int, Callable] = {}


_experiment_table_schema = """
CREATE  TABLE IF NOT EXISTS experiments (
    -- this will autoncrement by default if
    -- no value is specified on insert
    exp_id INTEGER PRIMARY KEY,
    name TEXT,
    sample_name TEXT,
    start_time INTEGER,
    end_time INTEGER,
    -- this is the last counter registered
    -- 1 based
    run_counter INTEGER,
    -- this is the formatter strin used to cosntruct
    -- the run name
    format_string TEXT
-- TODO: maybe I had a good reason for this doulbe primary key
--    PRIMARY KEY (exp_id, start_time, sample_name)
);
"""

_runs_table_schema = """
CREATE TABLE IF NOT EXISTS runs (
    -- this will autoincrement by default if
    -- no value is specified on insert
    run_id INTEGER PRIMARY KEY,
    exp_id INTEGER,
    -- friendly name for the run
    name TEXT,
    -- the name of the table which stores
    -- the actual results
    result_table_name TEXT,
    -- this is the run counter in its experiment 0 based
    result_counter INTEGER,
    ---
    run_timestamp INTEGER,
    completed_timestamp INTEGER,
    is_completed BOOL,
    parameters TEXT,
    -- metadata fields are added dynamically
    FOREIGN KEY(exp_id)
    REFERENCES
        experiments(exp_id)
);
"""

_layout_table_schema = """
CREATE TABLE IF NOT EXISTS layouts (
    layout_id INTEGER PRIMARY KEY,
    run_id INTEGER,
    -- name matching column name in result table
    parameter TEXT,
    label TEXT,
    unit TEXT,
    inferred_from TEXT,
    FOREIGN KEY(run_id)
    REFERENCES
        runs(run_id)
);
"""

_dependencies_table_schema = """
CREATE TABLE IF NOT EXISTS dependencies (
    dependent INTEGER,
    independent INTEGER,
    axis_num INTEGER
);
"""

_unicode_categories = ('Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nd', 'Pc', 'Pd', 'Zs')

# in the current version, these are the standard columns of the "runs" table
# Everything else is metadata
RUNS_TABLE_COLUMNS = ["run_id", "exp_id", "name", "result_table_name",
                      "result_counter", "run_timestamp", "completed_timestamp",
                      "is_completed", "parameters", "guid",
                      "run_description"]


def sql_placeholder_string(n: int) -> str:
    """
    Return an SQL value placeholder string for n values.
    Example: sql_placeholder_string(5) returns '(?,?,?,?,?)'
    """
    return '(' + ','.join('?'*n) + ')'


class ConnectionPlus(wrapt.ObjectProxy):
    """
    A class to extend the sqlite3.Connection object. Since sqlite3.Connection
    has no __dict__, we can not directly add attributes to its instance
    directly.

    It is not allowed to instantiate a new `ConnectionPlus` object from a
    `ConnectionPlus` object.

    Attributes:
        atomic_in_progress: a bool describing whether the connection is
            currently in the middle of an atomic block of transactions, thus
            allowing to nest `atomic` context managers
    """
    atomic_in_progress: bool = False

    def __init__(self, sqlite3_connection: sqlite3.Connection):
        super(ConnectionPlus, self).__init__(sqlite3_connection)

        if isinstance(sqlite3_connection, ConnectionPlus):
            raise ValueError('Attempted to create `ConnectionPlus` from a '
                             '`ConnectionPlus` object which is not allowed.')


def upgrader(func: Callable[[ConnectionPlus], None]):
    """
    Decorator for database version upgrade functions. An upgrade function
    must have the name `perform_db_upgrade_N_to_M` where N = M-1. For
    simplicity, an upgrade function must take a single argument of type
    `ConnectionPlus`. The upgrade function must either perform the upgrade
    and return (no return values allowed) or fail to perform the upgrade,
    in which case it must raise a RuntimeError. A failed upgrade must be
    completely rolled back before the RuntimeError is raises.

    The decorator takes care of logging about the upgrade and managing the
    database versioning.
    """
    name_comps = func.__name__.split('_')
    if not len(name_comps) == 6:
        raise NameError('Decorated function not a valid upgrader. '
                        'Must have name "perform_db_upgrade_N_to_M"')
    if not ''.join(name_comps[:3]+[name_comps[4]]) == 'performdbupgradeto':
        raise NameError('Decorated function not a valid upgrader. '
                        'Must have name "perform_db_upgrade_N_to_M"')
    from_version = int(name_comps[3])
    to_version = int(name_comps[5])

    if not to_version == from_version+1:
        raise ValueError(f'Invalid upgrade versions in function name: '
                         f'{func.__name__}; upgrade from version '
                         f'{from_version} to version {to_version}.'
                         ' Can only upgrade from version N'
                         ' to version N+1')

    @wraps(func)
    def do_upgrade(conn: ConnectionPlus) -> None:

        log.info(f'Starting database upgrade version {from_version} '
                 f'to {to_version}')

        start_version = get_user_version(conn)
        if start_version != from_version:
            log.info(f'Skipping upgrade {from_version} -> {to_version} as'
                     f' current database version is {start_version}.')
            return

        # This function either raises or returns
        func(conn)

        set_user_version(conn, to_version)
        log.info(f'Succesfully performed upgrade {from_version} '
                 f'-> {to_version}')

    _UPGRADE_ACTIONS[to_version] = do_upgrade

    return do_upgrade


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
    gets converted to `np.nan`.
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

    # If it is not 'nan', then we need to see if the value is really an
    # integer or with floating point digits
    numeric_int = int(numeric)
    if numeric != numeric_int:
        return numeric
    else:
        return numeric_int


def _adapt_float(fl: float) -> Union[float, str]:
    if np.isnan(fl):
        return "nan"
    return float(fl)


def one(curr: sqlite3.Cursor, column: Union[int, str]) -> Any:
    """Get the value of one column from one row
    Args:
        curr: cursor to operate on
        column: name of the column

    Returns:
        the value
    """
    res = curr.fetchall()
    if len(res) > 1:
        raise RuntimeError("Expected only one row")
    elif len(res) == 0:
        raise RuntimeError("Expected one row")
    else:
        return res[0][column]


def many(curr: sqlite3.Cursor, *columns: str) -> List[Any]:
    """Get the values of many columns from one row
    Args:
        curr: cursor to operate on
        columns: names of the columns

    Returns:
        list of  values
    """
    res = curr.fetchall()
    if len(res) > 1:
        raise RuntimeError("Expected only one row")
    else:
        return [res[0][c] for c in columns]


def many_many(curr: sqlite3.Cursor, *columns: str) -> List[List[Any]]:
    """Get all values of many columns
    Args:
        curr: cursor to operate on
        columns: names of the columns

    Returns:
        list of lists of values
    """
    res = curr.fetchall()
    results = []
    for r in res:
        results.append([r[c] for c in columns])
    return results


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

    if debug:
        conn.set_trace_callback(print)

    init_db(conn)
    perform_db_upgrade(conn, version=version)
    return conn


def perform_db_upgrade(conn: ConnectionPlus, version: int=-1) -> None:
    """
    This is intended to perform all upgrades as needed to bring the
    db from version 0 to the most current version (or the version specified).
    All the perform_db_upgrade_X_to_Y functions must raise if they cannot
    upgrade and be a NOOP if the current version is higher than their target.

    Args:
        conn: object for connection to the database
        version: Which version to upgrade to. We count from 0. -1 means
          'newest version'
    """
    version = _latest_available_version() if version == -1 else version

    current_version = get_user_version(conn)
    if current_version < version:
        log.info("Commencing database upgrade")
        for target_version in sorted(_UPGRADE_ACTIONS)[:version]:
            _UPGRADE_ACTIONS[target_version](conn)


@upgrader
def perform_db_upgrade_0_to_1(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 0 to version 1

    Add a GUID column to the runs table and assign guids for all existing runs
    """

    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
    cur = atomic_transaction(conn, sql)
    n_run_tables = len(cur.fetchall())

    if n_run_tables == 1:
        with atomic(conn) as conn:
            sql = "ALTER TABLE runs ADD COLUMN guid TEXT"
            transaction(conn, sql)
            # now assign GUIDs to existing runs
            cur = transaction(conn, 'SELECT run_id FROM runs')
            run_ids = [r[0] for r in many_many(cur, 'run_id')]

            for run_id in run_ids:
                query = f"""
                        SELECT run_timestamp
                        FROM runs
                        WHERE run_id == {run_id}
                        """
                cur = transaction(conn, query)
                timestamp = one(cur, 'run_timestamp')
                timeint = int(np.round(timestamp*1000))
                sql = f"""
                        UPDATE runs
                        SET guid = ?
                        where run_id == {run_id}
                        """
                sampleint = 3736062718  # 'deafcafe'
                cur.execute(sql, (generate_guid(timeint=timeint,
                                                sampleint=sampleint),))
    else:
        raise RuntimeError(f"found {n_run_tables} runs tables expected 1")


@upgrader
def perform_db_upgrade_1_to_2(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 1 to version 2

    Add two indeces on the runs table, one for exp_id and one for GUID
    """

    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
    cur = atomic_transaction(conn, sql)
    n_run_tables = len(cur.fetchall())

    if n_run_tables == 1:
        _IX_runs_exp_id = """
                          CREATE INDEX
                          IF NOT EXISTS IX_runs_exp_id
                          ON runs (exp_id DESC)
                          """
        _IX_runs_guid = """
                        CREATE INDEX
                        IF NOT EXISTS IX_runs_guid
                        ON runs (guid DESC)
                        """
        with atomic(conn) as conn:
            transaction(conn, _IX_runs_exp_id)
            transaction(conn, _IX_runs_guid)
    else:
        raise RuntimeError(f"found {n_run_tables} runs tables expected 1")


def _2to3_get_result_tables(conn: ConnectionPlus) -> Dict[int, str]:
    rst_query = "SELECT run_id, result_table_name FROM runs"
    cur = conn.cursor()
    cur.execute(rst_query)

    data = cur.fetchall()
    cur.close()
    results = {}
    for row in data:
        results[row['run_id']] = row['result_table_name']
    return results


def _2to3_get_layout_ids(conn: ConnectionPlus) -> DefaultDict[int, List[int]]:
    query = """
            select runs.run_id, layouts.layout_id
            FROM layouts
            INNER JOIN runs ON runs.run_id == layouts.run_id
            """
    cur = conn.cursor()
    cur.execute(query)
    data = cur.fetchall()
    cur.close()

    results: DefaultDict[int, List[int]] = defaultdict(list)

    for row in data:
        run_id = row['run_id']
        layout_id = row['layout_id']
        results[run_id].append(layout_id)

    return results


def _2to3_get_indeps(conn: ConnectionPlus) -> DefaultDict[int, List[int]]:
    query = """
            SELECT layouts.run_id, layouts.layout_id
            FROM layouts
            INNER JOIN dependencies
            ON layouts.layout_id==dependencies.independent
            """
    cur = conn.cursor()
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    results: DefaultDict[int, List[int]] = defaultdict(list)

    for row in data:
        run_id = row['run_id']
        layout_id = row['layout_id']
        results[run_id].append(layout_id)

    return results


def _2to3_get_deps(conn: ConnectionPlus) -> DefaultDict[int, List[int]]:
    query = """
            SELECT layouts.run_id, layouts.layout_id
            FROM layouts
            INNER JOIN dependencies
            ON layouts.layout_id==dependencies.dependent
            """
    cur = conn.cursor()
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    results: DefaultDict[int, List[int]] = defaultdict(list)

    for row in data:
        run_id = row['run_id']
        layout_id = row['layout_id']
        results[run_id].append(layout_id)

    return results


def _2to3_get_dependencies(conn: ConnectionPlus) -> DefaultDict[int, List[int]]:
    query = """
            SELECT dependent, independent
            FROM dependencies
            ORDER BY dependent, axis_num ASC
            """
    cur = conn.cursor()
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    results: DefaultDict[int, List[int]] = defaultdict(list)

    if len(data) == 0:
        return results

    for row in data:
        dep = row['dependent']
        indep = row['independent']
        results[dep].append(indep)

    return results


def _2to3_get_layouts(conn: ConnectionPlus) -> Dict[int,
                                                    Tuple[str, str, str, str]]:
    query = """
            SELECT layout_id, parameter, label, unit, inferred_from
            FROM layouts
            """
    cur = conn.cursor()
    cur.execute(query)

    results: Dict[int, Tuple[str, str, str, str]] = {}
    for row in cur.fetchall():
        results[row['layout_id']] = (row['parameter'],
                                     row['label'],
                                     row['unit'],
                                     row['inferred_from'])
    return results


def _2to3_get_paramspecs(conn: ConnectionPlus,
                         layout_ids: List[int],
                         layouts: Dict[int, Tuple[str, str, str, str]],
                         dependencies: Dict[int, List[int]],
                         deps: Sequence[int],
                         indeps: Sequence[int],
                         result_table_name: str) -> Dict[int, ParamSpec]:

    paramspecs: Dict[int, ParamSpec] = {}

    the_rest = set(layout_ids).difference(set(deps).union(set(indeps)))

    # We ensure that we first retrieve the ParamSpecs on which other ParamSpecs
    # depend, then the dependent ParamSpecs and finally the rest

    for layout_id in list(indeps) + list(deps) + list(the_rest):
        (name, label, unit, inferred_from_str) = layouts[layout_id]
        # get the data type
        sql = f'PRAGMA TABLE_INFO("{result_table_name}")'
        c = transaction(conn, sql)
        for row in c.fetchall():
            if row['name'] == name:
                paramtype = row['type']
                break

        inferred_from: List[str] = []
        depends_on: List[str] = []

        # this parameter depends on another parameter
        if layout_id in deps:
            setpoints = dependencies[layout_id]
            depends_on = [paramspecs[idp].name for idp in setpoints]

        if inferred_from_str != '':
            inferred_from = inferred_from_str.split(', ')

        paramspec = ParamSpec(name=name,
                              paramtype=paramtype,
                              label=label, unit=unit,
                              depends_on=depends_on,
                              inferred_from=inferred_from)
        paramspecs[layout_id] = paramspec

    return paramspecs


@upgrader
def perform_db_upgrade_2_to_3(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 2 to version 3

    Insert a new column, run_description, to the runs table and fill it out
    for exisitng runs with information retrieved from the layouts and
    dependencies tables represented as the to_json output of a RunDescriber
    object
    """

    no_of_runs_query = "SELECT max(run_id) FROM runs"
    no_of_runs = one(atomic_transaction(conn, no_of_runs_query), 'max(run_id)')
    no_of_runs = no_of_runs or 0

    # If one run fails, we want the whole upgrade to roll back, hence the
    # entire upgrade is one atomic transaction

    with atomic(conn) as conn:
        sql = "ALTER TABLE runs ADD COLUMN run_description TEXT"
        transaction(conn, sql)

        result_tables = _2to3_get_result_tables(conn)
        layout_ids_all = _2to3_get_layout_ids(conn)
        indeps_all = _2to3_get_indeps(conn)
        deps_all = _2to3_get_deps(conn)
        layouts = _2to3_get_layouts(conn)
        dependencies = _2to3_get_dependencies(conn)

        pbar = tqdm(range(1, no_of_runs+1))
        pbar.set_description("Upgrading database")

        for run_id in pbar:

            if run_id in layout_ids_all:

                result_table_name = result_tables[run_id]
                layout_ids = list(layout_ids_all[run_id])
                if run_id in indeps_all:
                    independents = tuple(indeps_all[run_id])
                else:
                    independents = ()
                if run_id in deps_all:
                    dependents = tuple(deps_all[run_id])
                else:
                    dependents = ()

                paramspecs = _2to3_get_paramspecs(conn,
                                                  layout_ids,
                                                  layouts,
                                                  dependencies,
                                                  dependents,
                                                  independents,
                                                  result_table_name)

                interdeps = InterDependencies(*paramspecs.values())
                desc = RunDescriber(interdeps=interdeps)
                json_str = desc.to_json()

            else:

                json_str = RunDescriber(InterDependencies()).to_json()

            sql = f"""
                   UPDATE runs
                   SET run_description = ?
                   WHERE run_id == ?
                   """
            cur = conn.cursor()
            cur.execute(sql, (json_str, run_id))
            log.debug(f"Upgrade in transition, run number {run_id}: OK")


@upgrader
def perform_db_upgrade_3_to_4(conn: ConnectionPlus) -> None:
    """
    Perform the upgrade from version 3 to version 4. This really
    repeats the version 3 upgrade as it originally had two bugs in
    the inferred annotation. inferred_from was passed incorrectly
    resulting in the parameter being marked inferred_from for each char
    in the inferred_from variable and inferred_from was not handled
    correctly for parameters that were neither dependencies nor dependent on
    other parameters. Both have since been fixed so rerun the upgrade.
    """

    no_of_runs_query = "SELECT max(run_id) FROM runs"
    no_of_runs = one(atomic_transaction(conn, no_of_runs_query), 'max(run_id)')
    no_of_runs = no_of_runs or 0

    # If one run fails, we want the whole upgrade to roll back, hence the
    # entire upgrade is one atomic transaction

    with atomic(conn) as conn:

        result_tables = _2to3_get_result_tables(conn)
        layout_ids_all = _2to3_get_layout_ids(conn)
        indeps_all = _2to3_get_indeps(conn)
        deps_all = _2to3_get_deps(conn)
        layouts = _2to3_get_layouts(conn)
        dependencies = _2to3_get_dependencies(conn)

        pbar = tqdm(range(1, no_of_runs+1))
        pbar.set_description("Upgrading database")

        for run_id in pbar:

            if run_id in layout_ids_all:

                result_table_name = result_tables[run_id]
                layout_ids = list(layout_ids_all[run_id])
                if run_id in indeps_all:
                    independents = tuple(indeps_all[run_id])
                else:
                    independents = ()
                if run_id in deps_all:
                    dependents = tuple(deps_all[run_id])
                else:
                    dependents = ()

                paramspecs = _2to3_get_paramspecs(conn,
                                                  layout_ids,
                                                  layouts,
                                                  dependencies,
                                                  dependents,
                                                  independents,
                                                  result_table_name)

                interdeps = InterDependencies(*paramspecs.values())
                desc = RunDescriber(interdeps=interdeps)
                json_str = desc.to_json()

            else:

                json_str = RunDescriber(InterDependencies()).to_json()

            sql = f"""
                   UPDATE runs
                   SET run_description = ?
                   WHERE run_id == ?
                   """
            cur = conn.cursor()
            cur.execute(sql, (json_str, run_id))
            log.debug(f"Upgrade in transition, run number {run_id}: OK")

def _latest_available_version() -> int:
    """Return latest available database schema version"""
    return len(_UPGRADE_ACTIONS)


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


def transaction(conn: ConnectionPlus,
                sql: str, *args: Any) -> sqlite3.Cursor:
    """Perform a transaction.
    The transaction needs to be committed or rolled back.


    Args:
        conn: database connection
        sql: formatted string
        *args: arguments to use for parameter substitution

    Returns:
        sqlite cursor

    """
    c = conn.cursor()
    if len(args) > 0:
        c.execute(sql, args)
    else:
        c.execute(sql)
    return c


def atomic_transaction(conn: ConnectionPlus,
                       sql: str, *args: Any) -> sqlite3.Cursor:
    """Perform an **atomic** transaction.
    The transaction is committed if there are no exceptions else the
    transaction is rolled back.
    NB: 'BEGIN' is by default only inserted before INSERT/UPDATE/DELETE/REPLACE
    but we want to guard any transaction that modifies the database (e.g. also
    ALTER). 'BEGIN' marks a place to commit from/roll back to

    Args:
        conn: database connection
        sql: formatted string
        *args: arguments to use for parameter substitution

    Returns:
        sqlite cursor

    """
    with atomic(conn) as atomic_conn:
        c = transaction(atomic_conn, sql, *args)
    return c


@contextmanager
def atomic(conn: ConnectionPlus):
    """
    Guard a series of transactions as atomic.

    If one transaction fails, all the previous transactions are rolled back
    and no more transactions are performed.

    NB: 'BEGIN' is by default only inserted before INSERT/UPDATE/DELETE/REPLACE
    but we want to guard any transaction that modifies the database (e.g. also
    ALTER)

    Args:
        conn: connection to guard
    """
    if not isinstance(conn, ConnectionPlus):
        raise ValueError('atomic context manager only accepts ConnectionPlus '
                         'database connection objects.')

    is_outmost = not(conn.atomic_in_progress)

    if conn.in_transaction and is_outmost:
        raise RuntimeError('SQLite connection has uncommitted transactions. '
                           'Please commit those before starting an atomic '
                           'transaction.')

    old_atomic_in_progress = conn.atomic_in_progress
    conn.atomic_in_progress = True

    try:
        if is_outmost:
            old_level = conn.isolation_level
            conn.isolation_level = None
            conn.cursor().execute('BEGIN')
        yield conn
    except Exception as e:
        conn.rollback()
        log.exception("Rolling back due to unhandled exception")
        raise RuntimeError("Rolling back due to unhandled exception") from e
    else:
        if is_outmost:
            conn.commit()
    finally:
        if is_outmost:
            conn.isolation_level = old_level
        conn.atomic_in_progress = old_atomic_in_progress


def make_connection_plus_from(conn: Union[sqlite3.Connection, ConnectionPlus]
                              ) -> ConnectionPlus:
    """
    Makes a ConnectionPlus connection object out of a given argument.

    If the given connection is already a ConnectionPlus, then it is returned
    without any changes.

    Args:
        conn: an sqlite database connection object

    Returns:
        the "same" connection but as ConnectionPlus object
    """
    if not isinstance(conn, ConnectionPlus):
        conn_plus = ConnectionPlus(conn)
    else:
        conn_plus = conn
    return conn_plus


def init_db(conn: ConnectionPlus)->None:
    with atomic(conn) as conn:
        transaction(conn, _experiment_table_schema)
        transaction(conn, _runs_table_schema)
        transaction(conn, _layout_table_schema)
        transaction(conn, _dependencies_table_schema)


def is_run_id_in_database(conn: ConnectionPlus,
                          *run_ids) -> Dict[int, bool]:
    """
    Look up run_ids and return a dictionary with the answers to the question
    "is this run_id in the database?"

    Args:
        conn: the connection to the database
        run_ids: the run_ids to look up

    Returns:
        a dict with the run_ids as keys and bools as values. True means that
        the run_id DOES exist in the database
    """
    run_ids = np.unique(run_ids)
    placeholders = sql_placeholder_string(len(run_ids))

    query = f"""
             SELECT run_id
             FROM runs
             WHERE run_id in {placeholders}
            """

    cursor = conn.cursor()
    cursor.execute(query, run_ids)
    rows = cursor.fetchall()
    existing_ids = [row[0] for row in rows]
    return {run_id: (run_id in existing_ids) for run_id in run_ids}


def is_column_in_table(conn: ConnectionPlus, table: str, column: str) -> bool:
    """
    A look-before-you-leap function to look up if a table has a certain column.

    Intended for the 'runs' table where columns might be dynamically added
    via `add_meta_data`/`insert_meta_data` functions.

    Args:
        conn: The connection
        table: the table name
        column: the column name
    """
    cur = atomic_transaction(conn, f"PRAGMA table_info({table})")
    for row in cur.fetchall():
        if row['name'] == column:
            return True
    return False


def insert_column(conn: ConnectionPlus, table: str, name: str,
                  paramtype: Optional[str] = None) -> None:
    """Insert new column to a table

    Args:
        conn: database connection
        table: destination for the insertion
        name: column name
        type: sqlite type of the column
    """
    # first check that the column is not already there
    # and do nothing if it is
    query = f'PRAGMA TABLE_INFO("{table}");'
    cur = atomic_transaction(conn, query)
    columns = many_many(cur, "name")
    if name in [col[0] for col in columns]:
        return

    with atomic(conn) as conn:
        if paramtype:
            transaction(conn,
                        f'ALTER TABLE "{table}" ADD COLUMN "{name}" '
                        f'{paramtype}')
        else:
            transaction(conn,
                        f'ALTER TABLE "{table}" ADD COLUMN "{name}"')


def select_one_where(conn: ConnectionPlus, table: str, column: str,
                     where_column: str, where_value: Any) -> Any:
    query = f"""
    SELECT {column}
    FROM
        {table}
    WHERE
        {where_column} = ?
    """
    cur = atomic_transaction(conn, query, where_value)
    res = one(cur, column)
    return res


def select_many_where(conn: ConnectionPlus, table: str, *columns: str,
                      where_column: str, where_value: Any) -> Any:
    _columns = ",".join(columns)
    query = f"""
    SELECT {_columns}
    FROM
        {table}
    WHERE
        {where_column} = ?
    """
    cur = atomic_transaction(conn, query, where_value)
    res = many(cur, *columns)
    return res


def _massage_dict(metadata: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    {key:value, key2:value} -> ["key=?, key2=?", [value, value]]
    """
    template = []
    values = []
    for key, value in metadata.items():
        template.append(f"{key} = ?")
        values.append(value)
    return ','.join(template), values


def update_where(conn: ConnectionPlus, table: str,
                 where_column: str, where_value: Any, **updates) -> None:
    _updates, values = _massage_dict(updates)
    query = f"""
    UPDATE
        '{table}'
    SET
        {_updates}
    WHERE
        {where_column} = ?
    """
    atomic_transaction(conn, query, *values, where_value)


def insert_values(conn: ConnectionPlus,
                  formatted_name: str,
                  columns: List[str],
                  values: VALUES,
                  ) -> int:
    """
    Inserts values for the specified columns.
    Will pad with null if not all parameters are specified.
    NOTE this need to be committed before closing the connection.
    """
    _columns = ",".join(columns)
    _values = ",".join(["?"] * len(columns))
    query = f"""INSERT INTO "{formatted_name}"
        ({_columns})
    VALUES
        ({_values})
    """

    c = atomic_transaction(conn, query, *values)
    return c.lastrowid


def insert_many_values(conn: ConnectionPlus,
                       formatted_name: str,
                       columns: List[str],
                       values: List[VALUES],
                       ) -> int:
    """
    Inserts many values for the specified columns.

    Example input:
    columns: ['xparam', 'yparam']
    values: [[x1, y1], [x2, y2], [x3, y3]]

    NOTE this need to be committed before closing the connection.
    """
    # We demand that all values have the same length
    lengths = [len(val) for val in values]
    if len(np.unique(lengths)) > 1:
        raise ValueError('Wrong input format for values. Must specify the '
                         'same number of values for all columns. Received'
                         f' lengths {lengths}.')
    no_of_rows = len(lengths)
    no_of_columns = lengths[0]

    # The TOTAL number of inserted values in one query
    # must be less than the SQLITE_MAX_VARIABLE_NUMBER

    # Version check cf.
    # "https://stackoverflow.com/questions/9527851/sqlite-error-
    #  too-many-terms-in-compound-select"
    version = qc.SQLiteSettings.settings['VERSION']

    # According to the SQLite changelog, the version number
    # to check against below
    # ought to be 3.7.11, but that fails on Travis
    if LooseVersion(str(version)) <= LooseVersion('3.8.2'):
        max_var = qc.SQLiteSettings.limits['MAX_COMPOUND_SELECT']
    else:
        max_var = qc.SQLiteSettings.limits['MAX_VARIABLE_NUMBER']
    rows_per_transaction = int(int(max_var)/no_of_columns)

    _columns = ",".join(columns)
    _values = "(" + ",".join(["?"] * len(values[0])) + ")"

    a, b = divmod(no_of_rows, rows_per_transaction)
    chunks = a*[rows_per_transaction] + [b]
    if chunks[-1] == 0:
        chunks.pop()

    start = 0
    stop = 0

    with atomic(conn) as conn:
        for ii, chunk in enumerate(chunks):
            _values_x_params = ",".join([_values] * chunk)

            query = f"""INSERT INTO "{formatted_name}"
                        ({_columns})
                        VALUES
                        {_values_x_params}
                     """
            stop += chunk
            # we need to make values a flat list from a list of list
            flattened_values = list(
                itertools.chain.from_iterable(values[start:stop]))

            c = transaction(conn, query, *flattened_values)

            if ii == 0:
                return_value = c.lastrowid
            start += chunk

    return return_value


def modify_values(conn: ConnectionPlus,
                  formatted_name: str,
                  index: int,
                  columns: List[str],
                  values: VALUES,
                  ) -> int:
    """
    Modify values for the specified columns.
    If a column is in the table but not in the columns list is
    left untouched.
    If a column is mapped to None, it will be a null value.
    """
    name_val_template = []
    for name in columns:
        name_val_template.append(f"{name}=?")
    name_val_templates = ",".join(name_val_template)
    query = f"""
    UPDATE "{formatted_name}"
    SET
        {name_val_templates}
    WHERE
        rowid = {index+1}
    """
    c = atomic_transaction(conn, query, *values)
    return c.rowcount


def modify_many_values(conn: ConnectionPlus,
                       formatted_name: str,
                       start_index: int,
                       columns: List[str],
                       list_of_values: List[VALUES],
                       ) -> None:
    """
    Modify many values for the specified columns.
    If a column is in the table but not in the column list is
    left untouched.
    If a column is mapped to None, it will be a null value.
    """
    _len = length(conn, formatted_name)
    len_requested = start_index + len(list_of_values[0])
    available = _len - start_index
    if len_requested > _len:
        reason = f"""Modify operation Out of bounds.
        Trying to modify {len(list_of_values)} results,
        but therere are only {available} results.
        """
        raise ValueError(reason)
    for values in list_of_values:
        modify_values(conn, formatted_name, start_index, columns, values)
        start_index += 1


def length(conn: ConnectionPlus,
           formatted_name: str
           ) -> int:
    """
    Return the lenght of the table

    Args:
        conn: the connection to the sqlite database
        formatted_name: name of the table

    Returns:
        the lenght of the table
    """
    query = f"select MAX(id) from '{formatted_name}'"
    c = atomic_transaction(conn, query)
    _len = c.fetchall()[0][0]
    if _len is None:
        return 0
    else:
        return _len


def _build_data_query( table_name: str,
             columns: List[str],
             start: Optional[int] = None,
             end: Optional[int] = None,
             ) -> str:

    _columns = ",".join(columns)
    query = f"""
            SELECT {_columns}
            FROM "{table_name}"
            """

    start_specified = start is not None
    end_specified = end is not None

    where = ' WHERE' if start_specified or end_specified else ''
    start_condition = f' rowid >= {start}' if start_specified else ''
    end_condition = f' rowid <= {end}' if end_specified else ''
    and_ = ' AND' if start_specified and end_specified else ''

    query += where + start_condition + and_ + end_condition
    return query


def get_data(conn: ConnectionPlus,
             table_name: str,
             columns: List[str],
             start: Optional[int] = None,
             end: Optional[int] = None,
             ) -> List[List[Any]]:
    """
    Get data from the columns of a table.
    Allows to specify a range of rows (1-based indexing, both ends are
    included).

    Args:
        conn: database connection
        table_name: name of the table
        columns: list of columns
        start: start of range; if None, then starts from the top of the table
        end: end of range; if None, then ends at the bottom of the table

    Returns:
        the data requested in the format of list of rows of values
    """
    if len(columns) == 0:
        warnings.warn(
            'get_data: requested data without specifying parameters/columns.'
            'Returning empty list.'
        )
        return [[]]
    query = _build_data_query(table_name, columns, start, end)
    c = atomic_transaction(conn, query)
    res = many_many(c, *columns)

    return res


def get_parameter_data(conn: ConnectionPlus,
                       table_name: str,
                       columns: Sequence[str]=(),
                       start: Optional[int]=None,
                       end: Optional[int]=None) -> \
        Dict[str, Dict[str, np.ndarray]]:
    """
    Get data for one or more parameters and its dependencies. The data
    is returned as numpy arrays within 2 layers of nested dicts. The keys of
    the outermost dict are the requested parameters and the keys of the second
    level are the loaded parameters (requested parameter followed by its
    dependencies). Start and End  sllows one to specify a range of rows
    (1-based indexing, both ends are included).

    Note that this assumes that all array type parameters have the same length.
    This should always be the case for a parameter and its dependencies.

    Note that all numeric data will at the moment be returned as floating point
    values.

    Args:
        conn: database connection
        table_name: name of the table
        columns: list of columns
        start: start of range; if None, then starts from the top of the table
        end: end of range; if None, then ends at the bottom of the table
    """
    sql = """
    SELECT run_id FROM runs WHERE result_table_name = ?
    """
    c = atomic_transaction(conn, sql, table_name)
    run_id = one(c, 'run_id')

    output = {}
    if len(columns) == 0:
        columns = get_non_dependencies(conn, run_id)

    # loop over all the requested parameters
    for output_param in columns:
        # find all the dependencies of this param
        paramspecs = get_parameter_dependencies(conn, output_param, run_id)
        param_names = [param.name for param in paramspecs]
        types = [param.type for param in paramspecs]

        res = get_data(conn, table_name, param_names, start=start, end=end)
        # if we have array type parameters expand all other parameters
        # to arrays
        if 'array' in types and ('numeric' in types or 'text' in types):
            first_array_element = types.index('array')
            numeric_elms = [i for i, x in enumerate(types)
                            if x == "numeric"]
            text_elms = [i for i, x in enumerate(types)
                         if x == "text"]
            for row in res:
                for element in numeric_elms:
                    row[element] = np.full_like(row[first_array_element],
                                                row[element],
                                                dtype=np.float)
                    # todo should we handle int/float types here
                    # we would in practice have to perform another
                    # loop to check that all elements of a given can be cast to
                    # int without loosing precision before choosing an integer
                    # representation of the array
                for element in text_elms:
                    strlen = len(row[element])
                    row[element] = np.full_like(row[first_array_element],
                                                row[element],
                                                dtype=f'U{strlen}')

        # Benchmarking shows that transposing the data with python types is
        # faster than transposing the data using np.array.transpose
        res_t = list(map(list, zip(*res)))
        output[output_param] = {name: np.array(column_data)
                         for name, column_data
                         in zip(param_names, res_t)}
    return output


def get_values(conn: ConnectionPlus,
               table_name: str,
               param_name: str) -> List[List[Any]]:
    """
    Get the not-null values of a parameter

    Args:
        conn: Connection to the database
        table_name: Name of the table that holds the data
        param_name: Name of the parameter to get the setpoints of

    Returns:
        The values
    """
    sql = f"""
    SELECT {param_name} FROM "{table_name}"
    WHERE {param_name} IS NOT NULL
    """
    c = atomic_transaction(conn, sql)
    res = many_many(c, param_name)

    return res


def get_setpoints(conn: ConnectionPlus,
                  table_name: str,
                  param_name: str) -> Dict[str, List[List[Any]]]:
    """
    Get the setpoints for a given dependent parameter

    Args:
        conn: Connection to the database
        table_name: Name of the table that holds the data
        param_name: Name of the parameter to get the setpoints of

    Returns:
        A list of returned setpoint values. Each setpoint return value
        is a list of lists of Any. The first list is a list of run points,
        the second list is a list of parameter values.
    """
    # TODO: We do this in no less than 5 table lookups, surely
    # this number can be reduced

    # get run_id
    sql = """
    SELECT run_id FROM runs WHERE result_table_name = ?
    """
    c = atomic_transaction(conn, sql, table_name)
    run_id = one(c, 'run_id')

    # get the parameter layout id
    sql = """
    SELECT layout_id FROM layouts
    WHERE parameter = ?
    and run_id = ?
    """
    c = atomic_transaction(conn, sql, param_name, run_id)
    layout_id = one(c, 'layout_id')

    # get the setpoint layout ids
    sql = """
    SELECT independent FROM dependencies
    WHERE dependent = ?
    """
    c = atomic_transaction(conn, sql, layout_id)
    indeps = many_many(c, 'independent')
    indeps = [idp[0] for idp in indeps]

    # get the setpoint names
    sql = f"""
    SELECT parameter FROM layouts WHERE layout_id
    IN {str(indeps).replace('[', '(').replace(']', ')')}
    """
    c = atomic_transaction(conn, sql)
    setpoint_names_temp = many_many(c, 'parameter')
    setpoint_names = [spn[0] for spn in setpoint_names_temp]
    setpoint_names = cast(List[str], setpoint_names)

    # get the actual setpoint data
    output: Dict[str, List[List[Any]]] = {}
    for sp_name in setpoint_names:
        sql = f"""
        SELECT {sp_name}
        FROM "{table_name}"
        WHERE {param_name} IS NOT NULL
        """
        c = atomic_transaction(conn, sql)
        sps = many_many(c, sp_name)
        output[sp_name] = sps

    return output


def get_runid_from_expid_and_counter(conn: ConnectionPlus, exp_id: int,
                                     counter: int) -> int:
    """
    Get the run_id of a run in the specified experiment with the specified
    counter

    Args:
        conn: connection to the database
        exp_id: the exp_id of the experiment containing the run
        counter: the intra-experiment run counter of that run
    """
    sql = """
          SELECT run_id
          FROM runs
          WHERE result_counter= ? AND
          exp_id = ?
          """
    c = transaction(conn, sql, counter, exp_id)
    run_id = one(c, 'run_id')
    return run_id


def get_runid_from_guid(conn: ConnectionPlus, guid: str) -> Union[int, None]:
    """
    Get the run_id of a run based on the guid

    Args:
        conn: connection to the database
        guid: the guid to look up

    Returns:
        The run_id if found, else -1.

    Raises:
        RuntimeError if more than one run with the given    GUID exists
    """
    query = """
            SELECT run_id
            FROM runs
            WHERE guid = ?
            """
    cursor = conn.cursor()
    cursor.execute(query, (guid,))
    rows = cursor.fetchall()
    if len(rows) == 0:
        run_id = -1
    elif len(rows) > 1:
        errormssg = ('Critical consistency error: multiple runs with'
                     f' the same GUID found! {len(rows)} runs have GUID '
                     f'{guid}')
        log.critical(errormssg)
        raise RuntimeError(errormssg)
    else:
        run_id = int(rows[0]['run_id'])

    return run_id


def get_layout(conn: ConnectionPlus,
               layout_id) -> Dict[str, str]:
    """
    Get the layout of a single parameter for plotting it

    Args:
        conn: The database connection
        layout_id: The run_id as in the layouts table

    Returns:
        A dict with name, label, and unit
    """
    sql = """
    SELECT parameter, label, unit FROM layouts WHERE layout_id=?
    """
    c = atomic_transaction(conn, sql, layout_id)
    t_res = many(c, 'parameter', 'label', 'unit')
    res = dict(zip(['name', 'label', 'unit'], t_res))
    return res


def get_layout_id(conn: ConnectionPlus,
                  parameter: Union[ParamSpec, str],
                  run_id: int) -> int:
    """
    Get the layout id of a parameter in a given run

    Args:
        conn: The database connection
        parameter: A ParamSpec or the name of the parameter
        run_id: The run_id of the run in question
    """
    # get the parameter layout id
    sql = """
    SELECT layout_id FROM layouts
    WHERE parameter = ?
    and run_id = ?
    """

    if isinstance(parameter, ParamSpec):
        name = parameter.name
    elif isinstance(parameter, str):
        name = parameter
    else:
        raise ValueError('Wrong parameter type, must be ParamSpec or str, '
                         f'received {type(parameter)}.')

    c = atomic_transaction(conn, sql, name, run_id)
    res = one(c, 'layout_id')

    return res


def get_dependents(conn: ConnectionPlus,
                   run_id: int) -> List[int]:
    """
    Get dependent layout_ids for a certain run_id, i.e. the layout_ids of all
    the dependent variables
    """
    sql = """
    SELECT layout_id FROM layouts
    WHERE run_id=? and layout_id in (SELECT dependent FROM dependencies)
    """
    c = atomic_transaction(conn, sql, run_id)
    res = [d[0] for d in many_many(c, 'layout_id')]
    return res


def get_dependencies(conn: ConnectionPlus,
                     layout_id: int) -> List[List[int]]:
    """
    Get the dependencies of a certain dependent variable (indexed by its
    layout_id)

    Args:
        conn: connection to the database
        layout_id: the layout_id of the dependent variable
    """
    sql = """
    SELECT independent, axis_num FROM dependencies WHERE dependent=?
    """
    c = atomic_transaction(conn, sql, layout_id)
    res = many_many(c, 'independent', 'axis_num')
    return res


def get_non_dependencies(conn: ConnectionPlus,
                         run_id: int) -> List[str]:
    """
    Return all parameters for a given run that are not dependencies of
    other parameters.

    Args:
        conn: connection to the database
        run_id: The run_id of the run in question

    Returns:
        A list of the parameter names.
    """
    parameters = get_parameters(conn, run_id)
    maybe_independent = []
    dependent = []
    dependencies: List[str] = []

    for param in parameters:
        if len(param.depends_on) == 0:
            maybe_independent.append(param.name)
        else:
            dependent.append(param.name)
            dependencies.extend(param.depends_on.split(', '))

    independent_set = set(maybe_independent) - set(dependencies)
    dependent_set = set(dependent)
    result = independent_set.union(dependent_set)
    return sorted(list(result))


# Higher level Wrappers


def get_parameter_dependencies(conn: ConnectionPlus, param: str,
                              run_id: int) -> List[ParamSpec]:
    """
    Given a parameter name return a list of ParamSpecs where the first
    element is the ParamSpec of the given parameter and the rest of the
    elements are ParamSpecs of its dependencies.

    Args:
        conn: connection to the database
        param: the name of the parameter to look up
        run_id: run_id: The run_id of the run in question

    Returns:
        List of ParameterSpecs of the parameter followed by its dependencies.
    """
    layout_id = get_layout_id(conn, param, run_id)
    deps = get_dependencies(conn, layout_id)
    parameters = [get_paramspec(conn, run_id, param)]

    for dep in deps:
        depinfo = get_layout(conn, dep[0])
        parameters.append(get_paramspec(conn, run_id, depinfo['name']))
    return parameters


def new_experiment(conn: ConnectionPlus,
                   name: str,
                   sample_name: str,
                   format_string: Optional[str]="{}-{}-{}",
                   start_time: Optional[float]=None,
                   end_time: Optional[float]=None,
                   ) -> int:
    """
    Add new experiment to container.

    Args:
        conn: database connection
        name: the name of the experiment
        sample_name: the name of the current sample
        format_string: basic format string for table-name
          must contain 3 placeholders.
        start_time: time when the experiment was started. Do not supply this
          unless you have a very good reason to do so.
        end_time: time when the experiment was completed. Do not supply this
          unless you have a VERY good reason to do so

    Returns:
        id: row-id of the created experiment
    """
    query = """
            INSERT INTO experiments
            (name, sample_name, format_string,
            run_counter, start_time, end_time)
            VALUES
            (?,?,?,?,?,?)
            """

    start_time = start_time or time.time()
    values = (name, sample_name, format_string, 0, start_time, end_time)

    curr = atomic_transaction(conn, query, *values)
    return curr.lastrowid


# TODO(WilliamHPNielsen): we should remove the redundant
# is_completed
def mark_run_complete(conn: ConnectionPlus, run_id: int):
    """ Mark run complete

    Args:
        conn: database connection
        run_id: id of the run to mark complete
        complete: wether the run is completed or not
    """
    query = """
    UPDATE
        runs
    SET
        completed_timestamp=?,
        is_completed=?
    WHERE run_id=?;
    """
    atomic_transaction(conn, query, time.time(), True, run_id)


def completed(conn: ConnectionPlus, run_id)->bool:
    """ Check if the run scomplete

    Args:
        conn: database connection
        run_id: id of the run to check
    """
    return bool(select_one_where(conn, "runs", "is_completed",
                                 "run_id", run_id))


def get_completed_timestamp_from_run_id(
        conn: ConnectionPlus, run_id: int) -> float:
    """
    Retrieve the timestamp when the given measurement run was completed

    If the measurement run has not been marked as completed, then the returned
    value is None.

    Args:
        conn: database connection
        run_id: id of the run

    Returns:
        timestamp in seconds since the Epoch, or None
    """
    return select_one_where(conn, "runs", "completed_timestamp",
                            "run_id", run_id)


def get_guid_from_run_id(conn: ConnectionPlus, run_id: int) -> str:
    """
    Get the guid of the given run

    Args:
        conn: database connection
        run_id: id of the run
    """
    return select_one_where(conn, "runs", "guid", "run_id", run_id)


def finish_experiment(conn: ConnectionPlus, exp_id: int):
    """ Finish experiment

    Args:
        conn: database connection
        name: the name of the experiment
    """
    query = """
    UPDATE experiments SET end_time=? WHERE exp_id=?;
    """
    atomic_transaction(conn, query, time.time(), exp_id)


def get_run_counter(conn: ConnectionPlus, exp_id: int) -> int:
    """ Get the experiment run counter

    Args:
        conn: the connection to the sqlite database
        exp_id: experiment identifier

    Returns:
        the exepriment run counter

    """
    return select_one_where(conn, "experiments", "run_counter",
                            where_column="exp_id",
                            where_value=exp_id)


def get_experiments(conn: ConnectionPlus) -> List[sqlite3.Row]:
    """ Get a list of experiments
     Args:
         conn: database connection

     Returns:
         list of rows
     """
    sql = """
    SELECT * FROM experiments
    """
    c = atomic_transaction(conn, sql)

    return c.fetchall()


def get_matching_exp_ids(conn: ConnectionPlus, **match_conditions) -> List:
    """
    Get exp_ids for experiments matching the match_conditions

    Raises:
        ValueError if a match_condition that is not "name", "sample_name",
        "format_string", "run_counter", "start_time", or "end_time"
    """
    valid_conditions = ["name", "sample_name", "start_time", "end_time",
                        "run_counter", "format_string"]

    for mcond in match_conditions:
        if mcond not in valid_conditions:
            raise ValueError(f"{mcond} is not a valid match condition.")

    end_time = match_conditions.get('end_time', None)
    time_eq = "=" if end_time is not None else "IS"

    sample_name = match_conditions.get('sample_name', None)
    sample_name_eq = "=" if sample_name is not None else "IS"

    query = "SELECT exp_id FROM experiments "
    for n, mcond in enumerate(match_conditions):
        if n == 0:
            query += f"WHERE {mcond} = ? "
        else:
            query += f"AND {mcond} = ? "

    # now some syntax clean-up
    if "format_string" in match_conditions:
        format_string = match_conditions["format_string"]
        query = query.replace("format_string = ?",
                              f'format_string = "{format_string}"')
        match_conditions.pop("format_string")
    query = query.replace("end_time = ?", f"end_time {time_eq} ?")
    query = query.replace("sample_name = ?", f"sample_name {sample_name_eq} ?")

    cursor = conn.cursor()
    cursor.execute(query, tuple(match_conditions.values()))
    rows = cursor.fetchall()

    return [row[0] for row in rows]


def get_exp_ids_from_run_ids(conn: ConnectionPlus,
                             run_ids: Sequence[int]) -> List[int]:
    """
    Get the corresponding exp_id for a sequence of run_ids

    Args:
        conn: connection to the database
        run_ids: a sequence of the run_ids to get the exp_id of

    Returns:
        A list of exp_ids matching the run_ids
    """
    sql_placeholders = sql_placeholder_string(len(run_ids))
    exp_id_query = f"""
                    SELECT exp_id
                    FROM runs
                    WHERE run_id IN {sql_placeholders}
                    """
    cursor = conn.cursor()
    cursor.execute(exp_id_query, run_ids)
    rows = cursor.fetchall()

    return [exp_id for row in rows for exp_id in row]


def get_last_experiment(conn: ConnectionPlus) -> Optional[int]:
    """
    Return last started experiment id

    Returns None if there are no experiments in the database
    """
    query = "SELECT MAX(exp_id) FROM experiments"
    c = atomic_transaction(conn, query)
    return c.fetchall()[0][0]


def get_runs(conn: ConnectionPlus,
             exp_id: Optional[int] = None)->List[sqlite3.Row]:
    """ Get a list of runs.

    Args:
        conn: database connection

    Returns:
        list of rows
    """
    with atomic(conn) as conn:
        if exp_id:
            sql = """
            SELECT * FROM runs
            where exp_id = ?
            """
            c = transaction(conn, sql, exp_id)
        else:
            sql = """
            SELECT * FROM runs
            """
            c = transaction(conn, sql)

    return c.fetchall()


def get_last_run(conn: ConnectionPlus, exp_id: int) -> Optional[int]:
    """
    Get run_id of the last run in experiment with exp_id

    Args:
        conn: connection to use for the query
        exp_id: id of the experiment to look inside

    Returns:
        the integer id of the last run or None if there are not runs in the
        experiment
    """
    query = """
    SELECT run_id, max(run_timestamp), exp_id
    FROM runs
    WHERE exp_id = ?;
    """
    c = atomic_transaction(conn, query, exp_id)
    return one(c, 'run_id')


def run_exists(conn: ConnectionPlus, run_id: int) -> bool:
    # the following query always returns a single sqlite3.Row with an integer
    # value of `1` or `0` for existing and non-existing run_id in the database
    query = """
    SELECT EXISTS(
        SELECT 1
        FROM runs
        WHERE run_id = ?
        LIMIT 1
    );
    """
    res: sqlite3.Row = atomic_transaction(conn, query, run_id).fetchone()
    return bool(res[0])


def data_sets(conn: ConnectionPlus) -> List[sqlite3.Row]:
    """ Get a list of datasets
    Args:
        conn: database connection

    Returns:
        list of rows
    """
    sql = """
    SELECT * FROM runs
    """
    c = atomic_transaction(conn, sql)
    return c.fetchall()


def format_table_name(fmt_str: str, name: str, exp_id: int,
                      run_counter: int) -> str:
    """
    Format the format_string into a table name

    Args:
        fmt_str: a valid format string
        name: the run name
        exp_id: the experiment ID
        run_counter: the intra-experiment runnumber of this run
    """
    table_name = fmt_str.format(name, exp_id, run_counter)
    _validate_table_name(table_name)  # raises if table_name not valid
    return table_name


def _insert_run(conn: ConnectionPlus, exp_id: int, name: str,
                guid: str,
                parameters: Optional[List[ParamSpec]] = None,
                ):
    # get run counter and formatter from experiments
    run_counter, format_string = select_many_where(conn,
                                                   "experiments",
                                                   "run_counter",
                                                   "format_string",
                                                   where_column="exp_id",
                                                   where_value=exp_id)
    run_counter += 1
    formatted_name = format_table_name(format_string, name, exp_id,
                                       run_counter)
    table = "runs"

    parameters = parameters or []
    desc_str = RunDescriber(InterDependencies(*parameters)).to_json()

    with atomic(conn) as conn:

        if parameters:
            query = f"""
            INSERT INTO {table}
                (name,
                 exp_id,
                 guid,
                 result_table_name,
                 result_counter,
                 run_timestamp,
                 parameters,
                 is_completed,
                 run_description)
            VALUES
                (?,?,?,?,?,?,?,?,?)
            """
            curr = transaction(conn, query,
                               name,
                               exp_id,
                               guid,
                               formatted_name,
                               run_counter,
                               time.time(),
                               ",".join([p.name for p in parameters]),
                               False,
                               desc_str)

            _add_parameters_to_layout_and_deps(conn, formatted_name,
                                               *parameters)

        else:
            query = f"""
            INSERT INTO {table}
                (name,
                 exp_id,
                 guid,
                 result_table_name,
                 result_counter,
                 run_timestamp,
                 is_completed,
                 run_description)
            VALUES
                (?,?,?,?,?,?,?,?)
            """
            curr = transaction(conn, query,
                               name,
                               exp_id,
                               guid,
                               formatted_name,
                               run_counter,
                               time.time(),
                               False,
                               desc_str)
    run_id = curr.lastrowid
    return run_counter, formatted_name, run_id


def _update_experiment_run_counter(conn: ConnectionPlus, exp_id: int,
                                   run_counter: int) -> None:
    query = """
    UPDATE experiments
    SET run_counter = ?
    WHERE exp_id = ?
    """
    atomic_transaction(conn, query, run_counter, exp_id)


def get_parameters(conn: ConnectionPlus,
                   run_id: int) -> List[ParamSpec]:
    """
    Get the list of param specs for run

    Args:
        conn: the connection to the sqlite database
        run_id: The id of the run

    Returns:
        A list of param specs for this run
    """

    sql = f"""
    SELECT parameter FROM layouts WHERE run_id={run_id}
    """
    c = conn.execute(sql)
    param_names_temp = many_many(c, 'parameter')
    param_names = [p[0] for p in param_names_temp]
    param_names = cast(List[str], param_names)

    parspecs = []

    for param_name in param_names:
        parspecs.append(get_paramspec(conn, run_id, param_name))

    return parspecs


def get_paramspec(conn: ConnectionPlus,
                  run_id: int,
                  param_name: str) -> ParamSpec:
    """
    Get the ParamSpec object for the given parameter name
    in the given run

    Args:
        conn: Connection to the database
        run_id: The run id
        param_name: The name of the parameter
    """

    # get table name
    sql = f"""
    SELECT result_table_name FROM runs WHERE run_id = {run_id}
    """
    c = conn.execute(sql)
    result_table_name = one(c, 'result_table_name')

    # get the data type
    sql = f"""
    PRAGMA TABLE_INFO("{result_table_name}")
    """
    c = conn.execute(sql)
    for row in c.fetchall():
        if row['name'] == param_name:
            param_type = row['type']
            break

    # get everything else

    sql = f"""
    SELECT * FROM layouts
    WHERE parameter="{param_name}" and run_id={run_id}
    """
    c = conn.execute(sql)
    resp = many(c, 'layout_id', 'run_id', 'parameter', 'label', 'unit',
                'inferred_from')
    (layout_id, _, _, label, unit, inferred_from_string) = resp

    if inferred_from_string:
        inferred_from = inferred_from_string.split(', ')
    else:
        inferred_from = []

    deps = get_dependencies(conn, layout_id)
    depends_on: Optional[List[str]]
    if len(deps) == 0:
        depends_on = None
    else:
        dps: List[int] = [dp[0] for dp in deps]
        ax_nums: List[int] = [dp[1] for dp in deps]
        depends_on = []
        for _, dp in sorted(zip(ax_nums, dps)):
            sql = f"""
            SELECT parameter FROM layouts WHERE layout_id = {dp}
            """
            c = conn.execute(sql)
            depends_on.append(one(c, 'parameter'))

    parspec = ParamSpec(param_name, param_type, label, unit,
                        inferred_from,
                        depends_on)
    return parspec


def update_run_description(conn: ConnectionPlus, run_id: int,
                           description: str) -> None:
    """
    Update the run_description field for the given run_id. The description
    string must be a valid JSON string representation of a RunDescriber object
    """
    try:
        RunDescriber.from_json(description)
    except Exception as e:
        raise ValueError("Invalid description string. Must be a JSON string "
                         "representaion of a RunDescriber object.") from e

    sql = """
          UPDATE runs
          SET run_description = ?
          WHERE run_id = ?
          """
    with atomic(conn) as conn:
        conn.cursor().execute(sql, (description, run_id))


def add_parameter(conn: ConnectionPlus,
                  formatted_name: str,
                  *parameter: ParamSpec):
    """
    Add parameters to the dataset

    This will update the layouts and dependencies tables

    NOTE: two parameters with the same name are not allowed

    Args:
        conn: the connection to the sqlite database
        formatted_name: name of the table
        parameter: the list of ParamSpecs for parameters to add
    """
    with atomic(conn) as conn:
        p_names = []
        for p in parameter:
            insert_column(conn, formatted_name, p.name, p.type)
            p_names.append(p.name)
        # get old parameters column from run table
        sql = f"""
        SELECT parameters FROM runs
        WHERE result_table_name=?
        """
        with atomic(conn) as conn:
            c = transaction(conn, sql, formatted_name)
        old_parameters = one(c, 'parameters')
        if old_parameters:
            new_parameters = ",".join([old_parameters] + p_names)
        else:
            new_parameters = ",".join(p_names)
        sql = "UPDATE runs SET parameters=? WHERE result_table_name=?"
        with atomic(conn) as conn:
            transaction(conn, sql, new_parameters, formatted_name)

        # Update the layouts table
        c = _add_parameters_to_layout_and_deps(conn, formatted_name,
                                               *parameter)


def _add_parameters_to_layout_and_deps(conn: ConnectionPlus,
                                       formatted_name: str,
                                       *parameter: ParamSpec) -> sqlite3.Cursor:
    # get the run_id
    sql = f"""
    SELECT run_id FROM runs WHERE result_table_name="{formatted_name}";
    """
    run_id = one(transaction(conn, sql), 'run_id')
    layout_args = []
    for p in parameter:
        layout_args.append(run_id)
        layout_args.append(p.name)
        layout_args.append(p.label)
        layout_args.append(p.unit)
        layout_args.append(p.inferred_from)
    rowplaceholder = '(?, ?, ?, ?, ?)'
    placeholder = ','.join([rowplaceholder] * len(parameter))
    sql = f"""
    INSERT INTO layouts (run_id, parameter, label, unit, inferred_from)
    VALUES {placeholder}
    """

    with atomic(conn) as conn:
        c = transaction(conn, sql, *layout_args)

        for p in parameter:

            if p.depends_on != '':

                layout_id = get_layout_id(conn, p, run_id)

                deps = p.depends_on.split(', ')
                for ax_num, dp in enumerate(deps):

                    sql = """
                    SELECT layout_id FROM layouts
                    WHERE run_id=? and parameter=?;
                    """

                    c = transaction(conn, sql, run_id, dp)
                    dep_ind = one(c, 'layout_id')

                    sql = """
                    INSERT INTO dependencies (dependent, independent, axis_num)
                    VALUES (?,?,?)
                    """

                    c = transaction(conn, sql, layout_id, dep_ind, ax_num)
    return c


def _validate_table_name(table_name: str) -> bool:
    valid = True
    for i in table_name:
        if unicodedata.category(i) not in _unicode_categories:
            valid = False
            raise RuntimeError("Invalid table name "
                               "{} starting at {}".format(table_name, i))
    return valid


def _create_run_table(conn: ConnectionPlus,
                      formatted_name: str,
                      parameters: Optional[List[ParamSpec]] = None,
                      values: Optional[VALUES] = None
                      ) -> None:
    """Create run table with formatted_name as name

    Args:
        conn: database connection
        formatted_name: the name of the table to create
    """
    _validate_table_name(formatted_name)

    with atomic(conn) as conn:

        if parameters and values:
            _parameters = ",".join([p.sql_repr() for p in parameters])
            query = f"""
            CREATE TABLE "{formatted_name}" (
                id INTEGER PRIMARY KEY,
                {_parameters}
            );
            """
            transaction(conn, query)
            # now insert values
            insert_values(conn, formatted_name,
                          [p.name for p in parameters], values)
        elif parameters:
            _parameters = ",".join([p.sql_repr() for p in parameters])
            query = f"""
            CREATE TABLE "{formatted_name}" (
                id INTEGER PRIMARY KEY,
                {_parameters}
            );
            """
            transaction(conn, query)
        else:
            query = f"""
            CREATE TABLE "{formatted_name}" (
                id INTEGER PRIMARY KEY
            );
            """
            transaction(conn, query)


def create_run(conn: ConnectionPlus, exp_id: int, name: str,
               guid: str,
               parameters: Optional[List[ParamSpec]]=None,
               values:  List[Any] = None,
               metadata: Optional[Dict[str, Any]]=None)->Tuple[int, int, str]:
    """ Create a single run for the experiment.


    This will register the run in the runs table, the counter in the
    experiments table and create a new table with the formatted name.

    Args:
        - conn: the connection to the sqlite database
        - exp_id: the experiment id we want to create the run into
        - name: a friendly name for this run
        - guid: the guid adhering to our internal guid format
        - parameters: optional list of parameters this run has
        - values:  optional list of values for the parameters
        - metadata: optional metadata dictionary

    Returns:
        - run_counter: the id of the newly created run (not unique)
        - run_id: the row id of the newly created run
        - formatted_name: the name of the newly created table
    """

    with atomic(conn):
        run_counter, formatted_name, run_id = _insert_run(conn,
                                                          exp_id,
                                                          name,
                                                          guid,
                                                          parameters)
        if metadata:
            add_meta_data(conn, run_id, metadata)
        _update_experiment_run_counter(conn, exp_id, run_counter)
        _create_run_table(conn, formatted_name, parameters, values)

    return run_counter, run_id, formatted_name


def get_metadata(conn: ConnectionPlus, tag: str, table_name: str):
    """ Get metadata under the tag from table
    """
    return select_one_where(conn, "runs", tag,
                            "result_table_name", table_name)


def get_metadata_from_run_id(conn: ConnectionPlus, run_id: int) -> Dict:
    """
    Get all metadata associated with the specified run
    """
    # TODO: promote snapshot to be present at creation time
    non_metadata = RUNS_TABLE_COLUMNS + ['snapshot']

    metadata = {}
    possible_tags = []

    # first fetch all columns of the runs table
    query = "PRAGMA table_info(runs)"
    cursor = conn.cursor()
    for row in cursor.execute(query):
        if row['name'] not in non_metadata:
            possible_tags.append(row['name'])

    # and then fetch whatever metadata the run might have
    for tag in possible_tags:
        query = f"""
                SELECT "{tag}"
                FROM runs
                WHERE run_id = ?
                AND "{tag}" IS NOT NULL
                """
        cursor.execute(query, (run_id,))
        row = cursor.fetchall()
        if row != []:
            metadata[tag] = row[0][tag]

    return metadata


def insert_meta_data(conn: ConnectionPlus, row_id: int, table_name: str,
                     metadata: Dict[str, Any]) -> None:
    """
    Insert new metadata column and add values. Note that None is not a valid
    metadata value

    Args:
        - conn: the connection to the sqlite database
        - row_id: the row to add the metadata at
        - table_name: the table to add to, defaults to runs
        - metadata: the metadata to add
    """
    for tag, val in metadata.items():
        if val is None:
            raise ValueError(f'Tag {tag} has value None. '
                             ' That is not a valid metadata value!')
    for key in metadata.keys():
        insert_column(conn, table_name, key)
    update_meta_data(conn, row_id, table_name, metadata)


def update_meta_data(conn: ConnectionPlus, row_id: int, table_name: str,
                     metadata: Dict[str, Any]) -> None:
    """
    Updates metadata (they must exist already)

    Args:
        - conn: the connection to the sqlite database
        - row_id: the row to add the metadata at
        - table_name: the table to add to, defaults to runs
        - metadata: the metadata to add
    """
    update_where(conn, table_name, 'rowid', row_id, **metadata)


def add_meta_data(conn: ConnectionPlus,
                  row_id: int,
                  metadata: Dict[str, Any],
                  table_name: str = "runs") -> None:
    """
    Add metadata data (updates if exists, create otherwise).
    Note that None is not a valid metadata value.

    Args:
        - conn: the connection to the sqlite database
        - row_id: the row to add the metadata at
        - metadata: the metadata to add
        - table_name: the table to add to, defaults to runs
    """
    try:
        insert_meta_data(conn, row_id, table_name, metadata)
    except sqlite3.OperationalError as e:
        # this means that the column already exists
        # so just insert the new value
        if str(e).startswith("duplicate"):
            update_meta_data(conn, row_id, table_name, metadata)
        else:
            raise e


def get_user_version(conn: ConnectionPlus) -> int:

    curr = atomic_transaction(conn, 'PRAGMA user_version')
    res = one(curr, 0)
    return res


def set_user_version(conn: ConnectionPlus, version: int) -> None:

    atomic_transaction(conn, 'PRAGMA user_version({})'.format(version))


def get_experiment_name_from_experiment_id(
        conn: ConnectionPlus, exp_id: int) -> str:
    return select_one_where(
        conn, "experiments", "name", "exp_id", exp_id)


def get_sample_name_from_experiment_id(
        conn: ConnectionPlus, exp_id: int) -> str:
    return select_one_where(
        conn, "experiments", "sample_name", "exp_id", exp_id)


def get_run_timestamp_from_run_id(conn: ConnectionPlus,
                                  run_id: int) -> float:
    return select_one_where(conn, "runs", "run_timestamp", "run_id", run_id)


def update_GUIDs(conn: ConnectionPlus) -> None:
    """
    Update all GUIDs in this database where either the location code or the
    work_station code is zero to use the location and work_station code from
    the qcodesrc.json file in home. Runs where it is not true that both codes
    are zero are skipped.
    """

    log.info('Commencing update of all GUIDs in database')

    cfg = qc.Config()

    location = cfg['GUID_components']['location']
    work_station = cfg['GUID_components']['work_station']

    if location == 0:
        log.warning('The location is still set to the default (0). Can not '
                    'proceed. Please configure the location before updating '
                    'the GUIDs.')
        return
    if work_station == 0:
        log.warning('The work_station is still set to the default (0). Can not'
                    ' proceed. Please configure the location before updating '
                    'the GUIDs.')
        return

    query = f"select MAX(run_id) from runs"
    c = atomic_transaction(conn, query)
    no_of_runs = c.fetchall()[0][0]

    # now, there are four actions we can take

    def _both_nonzero(run_id: int, *args) -> None:
        log.info(f'Run number {run_id} already has a valid GUID, skipping.')

    def _location_only_zero(run_id: int, *args) -> None:
        log.warning(f'Run number {run_id} has a zero (default) location '
                    'code, but a non-zero work station code. Please manually '
                    'resolve this, skipping the run now.')

    def _workstation_only_zero(run_id: int, *args) -> None:
        log.warning(f'Run number {run_id} has a zero (default) work station'
                    ' code, but a non-zero location code. Please manually '
                    'resolve this, skipping the run now.')

    def _both_zero(run_id: int, conn, guid_comps) -> None:
        guid_str = generate_guid(timeint=guid_comps['time'],
                                 sampleint=guid_comps['sample'])
        with atomic(conn) as conn:
            sql = f"""
                   UPDATE runs
                   SET guid = ?
                   where run_id == {run_id}
                   """
            cur = conn.cursor()
            cur.execute(sql, (guid_str,))

        log.info(f'Succesfully updated run number {run_id}.')

    actions: Dict[Tuple[bool, bool], Callable]
    actions = {(True, True): _both_zero,
               (False, True): _workstation_only_zero,
               (True, False): _location_only_zero,
               (False, False): _both_nonzero}

    for run_id in range(1, no_of_runs+1):
        guid_str = get_guid_from_run_id(conn, run_id)
        guid_comps = parse_guid(guid_str)
        loc = guid_comps['location']
        ws = guid_comps['work_station']

        log.info(f'Updating run number {run_id}...')
        actions[(loc == 0, ws == 0)](run_id, conn, guid_comps)


def remove_trigger(conn: ConnectionPlus, trigger_id: str) -> None:
    """
    Removes a trigger with a given id if it exists.

    Note that this transaction is not atomic!

    Args:
        conn: database connection object
        name: id of the trigger
    """
    transaction(conn, f"DROP TRIGGER IF EXISTS {trigger_id};")


def _fix_wrong_run_descriptions(conn: ConnectionPlus,
                                run_ids: Sequence[int]) -> None:
    """
    NB: This is a FIX function. Do not use it unless your database has been
    diagnosed with the problem that this function fixes.

    Overwrite faulty run_descriptions by using information from the layouts and
    dependencies tables. If a correct description is found for a run, that
    run is left untouched.

    Args:
        conn: The connection to the database
        run_ids: The runs to (potentially) fix
    """

    log.info('[*] Fixing run descriptions...')
    for run_id in run_ids:
        trusted_paramspecs = get_parameters(conn, run_id)
        trusted_desc = RunDescriber(
                           interdeps=InterDependencies(*trusted_paramspecs))

        actual_desc_str = select_one_where(conn, "runs",
                                           "run_description",
                                           "run_id", run_id)

        if actual_desc_str == trusted_desc.to_json():
            log.info(f'[+] Run id: {run_id} had an OK description')
        else:
            log.info(f'[-] Run id: {run_id} had a broken description. '
                     f'Description found: {actual_desc_str}')
            update_run_description(conn, run_id, trusted_desc.to_json())
            log.info(f'    Run id: {run_id} has been updated.')
