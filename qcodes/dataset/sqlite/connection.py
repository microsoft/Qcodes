"""
This module provides a wrapper class :class:`ConnectionPlus` around
:class:`sqlite3.Connection` together with functions around it which allow
performing nested atomic transactions on an SQLite database.
"""
import logging
import sqlite3
from contextlib import contextmanager
from typing import Union, Any, Iterator

import wrapt

from qcodes.utils.delaykeyboardinterrupt import DelayedKeyboardInterrupt

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
        path_to_dbfile: Path to the database file of the connection.
    """
    atomic_in_progress: bool = False
    path_to_dbfile = ''

    def __init__(self, sqlite3_connection: sqlite3.Connection):
        super(ConnectionPlus, self).__init__(sqlite3_connection)

        if isinstance(sqlite3_connection, ConnectionPlus):
            raise ValueError('Attempted to create `ConnectionPlus` from a '
                             '`ConnectionPlus` object which is not allowed.')

        self.path_to_dbfile = path_to_dbfile(sqlite3_connection)


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


@contextmanager
def atomic(conn: ConnectionPlus) -> Iterator[ConnectionPlus]:
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
    with DelayedKeyboardInterrupt():
        if not isinstance(conn, ConnectionPlus):
            raise ValueError('atomic context manager only accepts '
                             'ConnectionPlus database connection objects.')

        is_outmost = not(conn.atomic_in_progress)

        if conn.in_transaction and is_outmost:
            raise RuntimeError('SQLite connection has uncommitted '
                               'transactions. '
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


def path_to_dbfile(conn: Union[ConnectionPlus, sqlite3.Connection]) -> str:
    """
    Return the path of the database file that the conn object is connected to
    """
    # according to https://www.sqlite.org/pragma.html#pragma_database_list
    # the 3th element (1 indexed) is the path
    cursor = conn.cursor()
    cursor.execute("PRAGMA database_list")
    row = cursor.fetchall()[0]

    return row[2]
