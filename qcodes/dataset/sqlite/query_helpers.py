import sqlite3
from typing import List, Any, Union, Dict, Tuple

from qcodes.dataset.sqlite.connection import ConnectionPlus, atomic_transaction


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
