"""
This module provides a number of convenient general-purpose functions that
are useful for building more database-specific queries out of them.
"""
import itertools
import sqlite3
from distutils.version import LooseVersion
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray

from qcodes.dataset.sqlite.connection import (
    ConnectionPlus,
    atomic,
    atomic_transaction,
    transaction,
)
from qcodes.dataset.sqlite.settings import SQLiteSettings
from qcodes.utils.deprecate import deprecate

# represent the type of  data we can/want map to sqlite column
VALUE = Union[str, complex, List, ndarray, bool, None]
VALUES = Sequence[VALUE]


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


def select_one_where(
    conn: ConnectionPlus, table: str, column: str, where_column: str, where_value: VALUE
) -> VALUE:
    """
    Select a value from a given column given a match of a value in a
    different column. If the given matched row/column intersect is empty
    None will be returned.

    Args:
        conn: Connection to the db
        table: Table to look for values in
        column: Column to return value from
        where_column: Column to match on
        where_value: Value to match in where_column

    Returns:
        Value found
    raises:
        RuntimeError if not exactly match is found.
    """
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


def select_many_where(
    conn: ConnectionPlus,
    table: str,
    *columns: str,
    where_column: str,
    where_value: VALUE,
) -> VALUES:
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


def _massage_dict(metadata: Mapping[str, Any]) -> Tuple[str, List[Any]]:
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
                 where_column: str, where_value: Any, **updates: Any) -> None:
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
                       columns: Sequence[str],
                       values: Sequence[VALUES],
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
    version = SQLiteSettings.settings['VERSION']

    # According to the SQLite changelog, the version number
    # to check against below
    # ought to be 3.7.11, but that fails on Travis
    if LooseVersion(str(version)) <= LooseVersion('3.8.2'):
        max_var = SQLiteSettings.limits['MAX_COMPOUND_SELECT']
    else:
        max_var = SQLiteSettings.limits['MAX_VARIABLE_NUMBER']
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


@deprecate('Unused private method to be removed in a future version')
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


@deprecate('Unused private method to be removed in a future version')
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
    Return the length of the table

    Args:
        conn: the connection to the sqlite database
        formatted_name: name of the table

    Returns:
        the length of the table
    """
    query = f"select MAX(id) from '{formatted_name}'"
    c = atomic_transaction(conn, query)
    _len = c.fetchall()[0][0]
    if _len is None:
        return 0
    else:
        return _len


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


def sql_placeholder_string(n: int) -> str:
    """
    Return an SQL value placeholder string for n values.
    Example: sql_placeholder_string(5) returns '(?,?,?,?,?)'
    """
    return '(' + ','.join('?'*n) + ')'
