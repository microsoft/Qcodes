import sqlite3

import wrapt


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