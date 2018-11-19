import sqlite3

import pytest

from qcodes.dataset.sqlite_base import ConnectionPlus, make_plus_connection_from


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
