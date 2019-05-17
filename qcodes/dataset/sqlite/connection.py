import logging
import sqlite3
from contextlib import contextmanager
from typing import Union

import wrapt


log = logging.getLogger(__name__)


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
