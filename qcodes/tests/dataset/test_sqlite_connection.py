import sqlite3

import pytest

from qcodes.dataset.sqlite_base import ConnectionPlus, \
    make_plus_connection_from, atomic


def sqlite_conn_in_transaction(conn: sqlite3.Connection):
    assert isinstance(conn, sqlite3.Connection)
    assert True is conn.in_transaction
    assert None is conn.isolation_level
    return True


def plus_conn_in_transaction(conn: ConnectionPlus):
    assert isinstance(conn, ConnectionPlus)
    assert True is conn.atomic_in_progress
    assert None is conn.isolation_level
    assert True is conn.in_transaction
    return True


def sqlite_conn_is_idle(conn: sqlite3.Connection, isolation=None):
    assert isinstance(conn, sqlite3.Connection)
    assert False is conn.in_transaction
    assert isolation == conn.isolation_level
    return True


def plus_conn_is_idle(conn: ConnectionPlus, isolation=None):
    assert isinstance(conn, ConnectionPlus)
    # assert False is conn.atomic_in_progress <-- should it be?
    assert True is conn.atomic_in_progress
    assert isolation == conn.isolation_level
    assert False is conn.in_transaction
    return True


def test_connection_plus():
    sqlite_conn = sqlite3.connect(':memory:')
    plus_conn = ConnectionPlus(sqlite_conn)

    assert isinstance(plus_conn, ConnectionPlus)
    assert isinstance(plus_conn, sqlite3.Connection)
    # reason for the value of "True" here is unknown
    assert True is plus_conn.atomic_in_progress
    # assert False is plus_conn.atomic_in_progress <-- should it be?


@pytest.mark.parametrize(
    argnames='conn',
    argvalues=(sqlite3.connect(':memory:'),
               ConnectionPlus(sqlite3.connect(':memory:'))),
    ids=('sqlite3.Connection', 'ConnectionPlus')
)
def test_make_plus_connection_from(conn):
    plus_conn = make_plus_connection_from(conn)

    assert isinstance(plus_conn, ConnectionPlus)

    if isinstance(conn, ConnectionPlus):
        # make_plus_connection_from does not change this, hence it should be
        # equal to the value from `conn` (which is True, see ConnectionPlus)
        assert conn.atomic_in_progress is plus_conn.atomic_in_progress
    else:
        # make_plus_connection_from explicitly sets this to False
        assert False is plus_conn.atomic_in_progress


def test_atomic_on_outmost_sqlite_connection():
    sqlite_conn = sqlite3.connect(':memory:')
    isolation_level = sqlite_conn.isolation_level
    assert False is sqlite_conn.in_transaction

    with atomic(sqlite_conn) as atomic_conn:
        assert sqlite_conn_in_transaction(sqlite_conn)
        assert plus_conn_in_transaction(atomic_conn)

    assert sqlite_conn_is_idle(sqlite_conn, isolation_level)
    assert plus_conn_is_idle(atomic_conn, isolation_level)


def test_atomic_on_outmost_plus_connection():
    sqlite_conn = sqlite3.connect(':memory:')
    plus_conn = ConnectionPlus(sqlite_conn)

    atomic_in_progress = plus_conn.atomic_in_progress

    isolation_level = plus_conn.isolation_level
    assert False is plus_conn.in_transaction

    with atomic(plus_conn) as atomic_conn:
        assert isinstance(atomic_conn, ConnectionPlus)

        assert True is atomic_conn.atomic_in_progress

        # assert None is atomic_conn.isolation_level <-- should it be?
        assert isolation_level == atomic_conn.isolation_level
        # assert None is plus_conn.isolation_level <-- should it be?
        assert isolation_level == plus_conn.isolation_level

        # assert True is plus_conn.in_transaction <-- should it be?
        assert False is plus_conn.in_transaction
        # assert True is atomic_conn.in_transaction <-- should it be?
        assert False is atomic_conn.in_transaction

    assert isolation_level == plus_conn.isolation_level
    assert False is plus_conn.in_transaction

    assert False is atomic_conn.in_transaction

    assert atomic_in_progress is atomic_conn.atomic_in_progress


def test_two_atomics_on_outmost_sqlite_connection():
    sqlite_conn = sqlite3.connect(':memory:')

    isolation_level = sqlite_conn.isolation_level
    assert False is sqlite_conn.in_transaction

    with atomic(sqlite_conn) as atomic_conn_1:
        assert sqlite_conn_in_transaction(sqlite_conn)
        assert plus_conn_in_transaction(atomic_conn_1)

        with atomic(atomic_conn_1) as atomic_conn_2:
            assert sqlite_conn_in_transaction(sqlite_conn)
            assert plus_conn_in_transaction(atomic_conn_2)
            assert plus_conn_in_transaction(atomic_conn_1)

        assert sqlite_conn_in_transaction(sqlite_conn)
        assert plus_conn_in_transaction(atomic_conn_1)
        assert plus_conn_in_transaction(atomic_conn_2)

    assert sqlite_conn_is_idle(sqlite_conn, isolation_level)
    assert plus_conn_is_idle(atomic_conn_1, isolation_level)
    assert plus_conn_is_idle(atomic_conn_2, isolation_level)


def test_two_atomics_on_outmost_plus_connection():
    sqlite_conn = sqlite3.connect(':memory:')
    plus_conn = ConnectionPlus(sqlite_conn)
    atomic_in_progress = plus_conn.atomic_in_progress

    isolation_level = plus_conn.isolation_level
    assert False is plus_conn.in_transaction

    with atomic(plus_conn) as atomic_conn_1:
        # assert plus_conn_in_transaction(plus_conn)
        assert plus_conn_is_idle(plus_conn, isolation_level)
        # assert plus_conn_in_transaction(atomic_conn_1)
        assert plus_conn_is_idle(atomic_conn_1, isolation_level)

        with atomic(atomic_conn_1) as atomic_conn_2:
            # assert plus_conn_in_transaction(plus_conn) <-- should it be?
            assert plus_conn_is_idle(plus_conn, isolation_level)
            # assert plus_conn_in_transaction(atomic_conn_1) <-- should it be?
            assert plus_conn_is_idle(atomic_conn_1, isolation_level)
            # assert plus_conn_in_transaction(atomic_conn_2) <-- should it be?
            assert plus_conn_is_idle(atomic_conn_2, isolation_level)

        # assert plus_conn_in_transaction(plus_conn) <-- should it be?
        assert plus_conn_is_idle(plus_conn, isolation_level)
        # assert plus_conn_in_transaction(atomic_conn_1) <-- should it be?
        assert plus_conn_is_idle(atomic_conn_1, isolation_level)
        # assert plus_conn_in_transaction(atomic_conn_2) <-- should it be?
        assert plus_conn_is_idle(atomic_conn_2, isolation_level)

    assert plus_conn_is_idle(plus_conn, isolation_level)
    assert plus_conn_is_idle(atomic_conn_1, isolation_level)
    assert plus_conn_is_idle(atomic_conn_2, isolation_level)

    assert atomic_in_progress == plus_conn.atomic_in_progress
    assert atomic_in_progress == atomic_conn_1.atomic_in_progress
    assert atomic_in_progress == atomic_conn_2.atomic_in_progress
