import re
import sqlite3

import pytest

from qcodes.dataset.sqlite.connection import (
    AtomicConnection,
    atomic,
    atomic_transaction,
)
from qcodes.dataset.sqlite.database import connect
from tests.common import error_caused_by


def sqlite_conn_in_transaction(conn: sqlite3.Connection):
    assert isinstance(conn, sqlite3.Connection)
    assert True is conn.in_transaction
    assert None is conn.isolation_level
    return True


def conn_plus_in_transaction(conn: AtomicConnection):
    assert isinstance(conn, AtomicConnection)
    assert True is conn.atomic_in_progress
    assert None is conn.isolation_level
    assert True is conn.in_transaction
    return True


def sqlite_conn_is_idle(conn: sqlite3.Connection, isolation=None):
    assert isinstance(conn, sqlite3.Connection)
    assert False is conn.in_transaction
    assert isolation == conn.isolation_level
    return True


def conn_plus_is_idle(conn: AtomicConnection, isolation=None):
    assert isinstance(conn, AtomicConnection)
    assert False is conn.atomic_in_progress
    assert isolation == conn.isolation_level
    assert False is conn.in_transaction
    return True


def test_atomic_connection() -> None:
    sqlite_conn = AtomicConnection(":memory:")

    assert sqlite_conn.path_to_dbfile == ""
    assert isinstance(sqlite_conn, AtomicConnection)
    assert isinstance(sqlite_conn, sqlite3.Connection)
    assert False is sqlite_conn.atomic_in_progress


def test_atomic_raises_for_non_atomic_conn() -> None:
    sqlite_conn = sqlite3.connect(":memory:")

    match_str = re.escape(
        "atomic context manager only accepts AtomicConnection "
        "database connection objects."
    )
    with pytest.raises(ValueError, match=match_str):
        with atomic(sqlite_conn):  # type: ignore[arg-type]
            pass


def test_atomic() -> None:
    conn_plus = AtomicConnection(":memory:")
    assert False is conn_plus.atomic_in_progress

    atomic_in_progress = conn_plus.atomic_in_progress
    isolation_level = conn_plus.isolation_level

    assert False is conn_plus.in_transaction

    with atomic(conn_plus) as atomic_conn:
        assert conn_plus_in_transaction(atomic_conn)
        assert conn_plus_in_transaction(conn_plus)

    assert isolation_level == conn_plus.isolation_level
    assert False is conn_plus.in_transaction
    assert atomic_in_progress is conn_plus.atomic_in_progress

    assert isolation_level == conn_plus.isolation_level
    assert False is atomic_conn.in_transaction
    assert atomic_in_progress is atomic_conn.atomic_in_progress


def test_atomic_with_exception() -> None:
    sqlite_conn = AtomicConnection(":memory:")

    sqlite_conn.execute("PRAGMA user_version(25)")
    sqlite_conn.commit()

    assert 25 == sqlite_conn.execute("PRAGMA user_version").fetchall()[0][0]

    with pytest.raises(
        RuntimeError, match="Rolling back due to unhandled exception"
    ) as e:
        with atomic(sqlite_conn) as atomic_conn:
            atomic_conn.execute("PRAGMA user_version(42)")
            raise Exception("intended exception")
    assert error_caused_by(e, "intended exception")

    assert 25 == sqlite_conn.execute("PRAGMA user_version").fetchall()[0][0]


def test_atomic_on_outmost_connection_that_is_in_transaction() -> None:
    conn = AtomicConnection(":memory:")

    conn.execute("BEGIN")
    assert True is conn.in_transaction

    match_str = re.escape(
        "SQLite connection has uncommitted transactions. "
        "Please commit those before starting an atomic "
        "transaction."
    )
    with pytest.raises(RuntimeError, match=match_str):
        with atomic(conn):
            pass


@pytest.mark.parametrize("in_transaction", (True, False))
def test_atomic_on_connection_plus_that_is_in_progress(in_transaction) -> None:
    sqlite_conn = AtomicConnection(":memory:")

    # explicitly set to True for testing purposes
    sqlite_conn.atomic_in_progress = True

    # implement parametrizing over connection's `in_transaction` attribute
    if in_transaction:
        sqlite_conn.cursor().execute("BEGIN")
    assert in_transaction is sqlite_conn.in_transaction

    isolation_level = sqlite_conn.isolation_level
    in_transaction = sqlite_conn.in_transaction

    with atomic(sqlite_conn) as atomic_conn:
        assert True is sqlite_conn.atomic_in_progress
        assert isolation_level == sqlite_conn.isolation_level
        assert in_transaction is sqlite_conn.in_transaction

        assert True is atomic_conn.atomic_in_progress
        assert isolation_level == atomic_conn.isolation_level
        assert in_transaction is atomic_conn.in_transaction

    assert True is sqlite_conn.atomic_in_progress
    assert isolation_level == sqlite_conn.isolation_level
    assert in_transaction is sqlite_conn.in_transaction

    assert True is atomic_conn.atomic_in_progress
    assert isolation_level == atomic_conn.isolation_level
    assert in_transaction is atomic_conn.in_transaction


def test_two_nested_atomics() -> None:
    sqlite_conn = AtomicConnection(":memory:")

    atomic_in_progress = sqlite_conn.atomic_in_progress
    isolation_level = sqlite_conn.isolation_level

    assert False is sqlite_conn.in_transaction

    with atomic(sqlite_conn) as atomic_conn_1:
        assert conn_plus_in_transaction(sqlite_conn)
        assert conn_plus_in_transaction(atomic_conn_1)

        with atomic(atomic_conn_1) as atomic_conn_2:
            assert conn_plus_in_transaction(sqlite_conn)
            assert conn_plus_in_transaction(atomic_conn_1)
            assert conn_plus_in_transaction(atomic_conn_2)

        assert conn_plus_in_transaction(sqlite_conn)
        assert conn_plus_in_transaction(atomic_conn_1)
        assert conn_plus_in_transaction(atomic_conn_2)

    assert conn_plus_is_idle(sqlite_conn, isolation_level)
    assert conn_plus_is_idle(atomic_conn_1, isolation_level)
    assert conn_plus_is_idle(atomic_conn_2, isolation_level)

    assert atomic_in_progress == sqlite_conn.atomic_in_progress
    assert atomic_in_progress == atomic_conn_1.atomic_in_progress
    assert atomic_in_progress == atomic_conn_2.atomic_in_progress


def test_that_use_of_atomic_commits_only_at_outermost_context(
    tmp_path,
) -> None:
    """
    This test tests the behavior of `AtomicConnection` that is created from
    `sqlite3.Connection` with respect to `atomic` context manager and commits.
    """
    dbfile = str(tmp_path / "temp.db")
    # just initialize the database file, connection objects needed for
    # testing in this test function are created separately, see below
    connect(dbfile)

    connection = AtomicConnection(dbfile)

    # this connection is going to be used to test whether changes have been
    # committed to the database file
    control_conn = connect(dbfile)

    get_all_runs = "SELECT * FROM runs"
    insert_run_with_name = "INSERT INTO runs (name) VALUES (?)"

    # assert that at the beginning of the test there are no runs in the
    # table; we'll be adding new rows to the runs table below

    assert 0 == len(connection.execute(get_all_runs).fetchall())
    assert 0 == len(control_conn.execute(get_all_runs).fetchall())

    # add 1 new row, and assert the state of the runs table at every step
    # note that control_conn will only detect the change after the `atomic`
    # context manager is exited

    with atomic(connection) as atomic_conn:
        assert 0 == len(connection.execute(get_all_runs).fetchall())
        assert 0 == len(atomic_conn.execute(get_all_runs).fetchall())
        assert 0 == len(control_conn.execute(get_all_runs).fetchall())

        atomic_conn.cursor().execute(insert_run_with_name, ["aaa"])

        assert 1 == len(connection.execute(get_all_runs).fetchall())
        assert 1 == len(atomic_conn.execute(get_all_runs).fetchall())
        assert 0 == len(control_conn.execute(get_all_runs).fetchall())

    assert 1 == len(connection.execute(get_all_runs).fetchall())
    assert 1 == len(atomic_conn.execute(get_all_runs).fetchall())
    assert 1 == len(control_conn.execute(get_all_runs).fetchall())

    # let's add two new rows but each inside its own `atomic` context manager
    # we expect to see the actual change in the database only after we exit
    # the outermost context.

    with atomic(connection) as atomic_conn_1:
        assert 1 == len(connection.execute(get_all_runs).fetchall())
        assert 1 == len(atomic_conn_1.execute(get_all_runs).fetchall())
        assert 1 == len(control_conn.execute(get_all_runs).fetchall())

        atomic_conn_1.cursor().execute(insert_run_with_name, ["bbb"])

        assert 2 == len(connection.execute(get_all_runs).fetchall())
        assert 2 == len(atomic_conn_1.execute(get_all_runs).fetchall())
        assert 1 == len(control_conn.execute(get_all_runs).fetchall())

        with atomic(atomic_conn_1) as atomic_conn_2:
            assert 2 == len(connection.execute(get_all_runs).fetchall())
            assert 2 == len(atomic_conn_1.execute(get_all_runs).fetchall())
            assert 2 == len(atomic_conn_2.execute(get_all_runs).fetchall())
            assert 1 == len(control_conn.execute(get_all_runs).fetchall())

            atomic_conn_2.cursor().execute(insert_run_with_name, ["ccc"])

            assert 3 == len(connection.execute(get_all_runs).fetchall())
            assert 3 == len(atomic_conn_1.execute(get_all_runs).fetchall())
            assert 3 == len(atomic_conn_2.execute(get_all_runs).fetchall())
            assert 1 == len(control_conn.execute(get_all_runs).fetchall())

        assert 3 == len(connection.execute(get_all_runs).fetchall())
        assert 3 == len(atomic_conn_1.execute(get_all_runs).fetchall())
        assert 3 == len(atomic_conn_2.execute(get_all_runs).fetchall())
        assert 1 == len(control_conn.execute(get_all_runs).fetchall())

    assert 3 == len(connection.execute(get_all_runs).fetchall())
    assert 3 == len(atomic_conn_1.execute(get_all_runs).fetchall())
    assert 3 == len(atomic_conn_2.execute(get_all_runs).fetchall())
    assert 3 == len(control_conn.execute(get_all_runs).fetchall())


def test_atomic_transaction(tmp_path) -> None:
    """Test that atomic_transaction works for AtomicConnection"""
    dbfile = str(tmp_path / "temp.db")

    conn = AtomicConnection(dbfile)

    sql_create_table = "CREATE TABLE smth (name TEXT)"

    atomic_transaction(conn, sql_create_table)

    ctrl_conn = sqlite3.connect(dbfile)
    sql_table_exists = 'SELECT sql FROM sqlite_master WHERE TYPE = "table"'
    assert sql_create_table in ctrl_conn.execute(sql_table_exists).fetchall()[0]


def test_atomic_transaction_on_sqlite3_connection_raises(tmp_path) -> None:
    """Test that atomic_transaction does not work for sqlite3.Connection"""
    dbfile = str(tmp_path / "temp.db")

    conn = sqlite3.connect(dbfile)

    match_str = re.escape(
        "atomic context manager only accepts AtomicConnection "
        "database connection objects."
    )

    with pytest.raises(ValueError, match=match_str):
        atomic_transaction(conn, "whatever sql query")  # type: ignore[arg-type]


def test_connect() -> None:
    conn = connect(":memory:")

    assert isinstance(conn, sqlite3.Connection)
    assert isinstance(conn, AtomicConnection)
    assert False is conn.atomic_in_progress
    assert None is conn.row_factory
