"""
This module provides a number of convenient general-purpose functions that
are useful for building more database-specific queries out of them.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from numpy import ndarray
from packaging import version

from qcodes.dataset.sqlite.connection import (
    ConnectionPlus,
    atomic,
    atomic_transaction,
    transaction,
)
from qcodes.dataset.sqlite.settings import SQLiteSettings

if TYPE_CHECKING:
    import sqlite3

# represent the type of  data we can/want map to sqlite column
VALUE = Union[str, complex, list, ndarray, bool, None]
VALUES = Sequence[VALUE]


def get_description_map(curr: sqlite3.Cursor) -> dict[str, int]:
    """Get the description of the last query
    Args:
        curr: last cursor operated on

    Returns:
        dictionary mapping column names and their indices
    """
    return {c[0]: i for i, c in enumerate(curr.description)}


def one(curr: sqlite3.Cursor, column: int | str) -> Any:
    """Get the value of one column from one row

    Args:
        curr: cursor to operate on
        column: name of the column or the index of the desired column in the
            result rows that ``cursor.fetchall()`` returns. In case a
            column name is being passed, it is important that the casing
            of the column name is exactly the one used when the column was
            created.

    Returns:
        the value
    """
    res = curr.fetchall()
    if len(res) > 1:
        raise RuntimeError("Expected only one row")
    elif len(res) == 0:
        raise RuntimeError("Expected one row")
    else:
        row = res[0]

        if isinstance(column, int):
            column_index_in_the_row = column
        else:
            columns_name_to_index_map = get_description_map(curr)
            maybe_column_index = columns_name_to_index_map.get(column)
            if maybe_column_index is not None:
                column_index_in_the_row = maybe_column_index
            else:
                # Note that the error message starts the same way as an
                # sqlite3 error about a column not existing:
                # ``no such column: <column name>`` - this is on purpose,
                # because if the given column name is not found in the
                # description map from the cursor then something is
                # definitely wrong, and likely the requested column does not
                # exist or the casing of its name is not equal to the casing
                # in the database
                raise RuntimeError(
                    f"no such column: {column}. "
                    f"Valid columns are {tuple(columns_name_to_index_map.keys())}"
                )

        return row[column_index_in_the_row]


def _need_to_select(curr: sqlite3.Cursor, *columns: str) -> bool:
    """
    Return True if the columns' description of the last query doesn't exactly match,
    the order is important
    """
    return tuple(c[0] for c in curr.description) != columns


def many(curr: sqlite3.Cursor, *columns: str) -> tuple[Any, ...]:
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
    elif _need_to_select(curr, *columns):
        raise RuntimeError(
            "Expected consistent selection: cursor has columns "
            f"{tuple(c[0] for c in curr.description)} but expected {columns}"
        )
    else:
        return res[0]


def many_many(curr: sqlite3.Cursor, *columns: str) -> list[tuple[Any, ...]]:
    """Get all values of many columns
    Args:
        curr: cursor to operate on
        columns: names of the columns

    Returns:
        list of lists of values
    """
    res = curr.fetchall()

    if _need_to_select(curr, *columns):
        raise RuntimeError(
            "Expected consistent selection: cursor has columns "
            f"{tuple(c[0] for c in curr.description)} but expected {columns}"
        )

    return res


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
        column: Column to return value from, it is important that the casing
            of the column name is exactly the one used when the column was
            created
        where_column: Column to match on
        where_value: Value to match in where_column

    Returns:
        Value found

    Raises:
        RuntimeError if not exactly one match is found.
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


def _massage_dict(metadata: Mapping[str, Any]) -> tuple[str, list[Any]]:
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


def insert_values(
    conn: ConnectionPlus,
    formatted_name: str,
    columns: list[str],
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
    return_value = c.lastrowid
    if return_value is None:
        raise RuntimeError(f"Insert_values into {formatted_name} failed")
    return return_value


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
    version_str = SQLiteSettings.settings["VERSION"]

    # According to the SQLite changelog, the version number
    # to check against below
    # ought to be 3.7.11, but that fails on Travis
    if version.parse(str(version_str)) <= version.parse("3.8.2"):
        max_var = SQLiteSettings.limits["MAX_COMPOUND_SELECT"]
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

    return_value = None
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

    if return_value is None:
        raise RuntimeError(f"insert_many_values into {formatted_name} failed ")
    return return_value


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
    # we replace ' in the table name to '' to make sure that
    # if the formatted name contains ' that will not cause the ' '
    # around formatted name to be ended
    # https://stackoverflow.com/questions/603572/escape-single-quote-character-for-use-in-an-sqlite-query
    escaped_formatted_name = formatted_name.replace("'", "''")
    query = f"select MAX(id) from '{escaped_formatted_name}'"
    c = atomic_transaction(conn, query)
    _len = c.fetchall()[0][0]
    if _len is None:
        return 0
    else:
        return _len


def insert_column(
    conn: ConnectionPlus, table: str, name: str, paramtype: str | None = None
) -> None:
    """Insert new column to a table

    Args:
        conn: database connection
        table: destination for the insertion
        name: column name
        paramtype: sqlite type of the column
    """
    # first check that the column is not already there
    # and do nothing if it is
    query = f'PRAGMA TABLE_INFO("{table}");'
    cur = atomic_transaction(conn, query)
    description = get_description_map(cur)
    columns = cur.fetchall()
    if name in [col[description["name"]] for col in columns]:
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
    description = get_description_map(cur)
    for row in cur.fetchall():
        if row[description["name"]] == column:
            return True
    return False


def sql_placeholder_string(n: int) -> str:
    """
    Return an SQL value placeholder string for n values.
    Example: sql_placeholder_string(5) returns '(?,?,?,?,?)'
    """
    return '(' + ','.join('?'*n) + ')'
