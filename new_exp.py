#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 unga <giulioungaretti@me.com>
#
# Distributed under terms of the MIT license.
"""
Add experiment
"""
import logging
import sqlite3
import time
from typing import Any, Optional, List


db = "/Users/unga/Desktop/experiment.db"


def one(curr: sqlite3.Cursor, column: str)->Any:
    res = curr.fetchall()
    if len(res) > 1:
        raise RuntimeError("Expected only one result")
    else:
        return res[0][column]


def many(curr: sqlite3.Cursor, *columns: str)->List[Any]:
    res = curr.fetchall()
    if len(res) > 1:
        raise RuntimeError("Expected only one result")
    else:
        return [res[0][c] for c in columns]


def connect(name: str, debug: bool=True) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    if debug:
        conn.set_trace_callback(print)
    return conn


def atomicTransaction(conn: sqlite3.Connection,
                      sql: str, *args: Any)->sqlite3.Cursor:
    c = conn.cursor()
    try:
        if len(args) > 0:
            c.execute(sql, args)
        else:
            c.execute(sql)
    except Exception as e:
        logging.exception("Could not  execute transaction, rolling back")
        conn.rollback()
    conn.commit()
    return c


def insert_column(conn: sqlite3.Connection, table: str, name: str,
                  type: str)->None:
    atomicTransaction(conn,
                      f"ALTER TABLE {table} ADD COLUMN {name} {type}",
                      None)


def new_experiment(conn: sqlite3.Connection,
                   name: str,
                   sample_name: str,
                   format_string: Optional[str] = "{}-{}"
                   )->int:
    """ Add new experiment to container

    Args:
        conn: database
        name: the name of the experiment
        sample_name: the name of the current sample
        format_string: TODO: write this
    """
    query = """
    INSERT INTO experiments
        (name, sample_name, start_time, format_string, run_counter)
    VALUES
        (?,?,?,?,?)
    """
    curr = atomicTransaction(conn, query, name, sample_name,
                             time.time(), format_string, 0)
    return curr.lastrowid


def finish_experiment(conn: sqlite3.Connection, exp_id: int):
    """ Finish experimen

    Args:
        conn: database
        name: the name of the experiment
    """
    query = """
    UPDATE experiments SET end_time=? WHERE exp_id=?;
    """
    atomicTransaction(conn, query, time.time(), exp_id)


def _select_one_where(conn: sqlite3.Connection, table: str, column: str,
                      where_column: str, where_value: Any) -> Any:
    query = f"""
    SELECT {column}
    FROM
        experiments
    WHERE
        {where_column} = ?
    """
    cur = atomicTransaction(conn, query, where_value)
    res = one(cur, column)
    return res


def _select_many_where(conn: sqlite3.Connection, table: str, *columns: str,
                       where_column: str, where_value: Any) -> Any:
    _columns = ",".join(columns)
    query = f"""
    SELECT {_columns}
    FROM
        experiments
    WHERE
        {where_column} = ?
    """
    cur = atomicTransaction(conn, query, where_value)
    res = many(cur, *columns)
    return res


# TODO: can make many of those. Easier to use // enforce some types
# but slower because make one query only
def get_run_counter(conn: sqlite3.Connection, exp_id: int) -> int:
    return _select_one_where(conn, "expereiments", "run_counter",
                             "exp_id", exp_id)


def create_run(conn: sqlite3.Connection, exp_id: int, name: str,
               parameters, metadata):
    # get run counter and formatter from experiments
    run_counter, format_string = _select_many_where(conn,
                                                    "experiments",
                                                    "run_counter",
                                                    "format_string",
                                                    where_column="exp_id",
                                                    where_value=exp_id)
    run_counter += 1
    formatted_name = format_string.format(name, run_counter)
    table = "runs"
    query = f"""
    INSERT INTO {table}
        (name,exp_id,result_table_name,result_counter, run_timestamp)
    VALUES
        (?,?,?,?,?)
    """
    curr = atomicTransaction(conn, query,
                             name,
                             exp_id,
                             formatted_name,
                             run_counter,
                             time.time()
                             )
    query = """
    UPDATE experiments
    SET run_counter = ?
    WHERE exp_id = ?
    """
    atomicTransaction(conn, query, run_counter, exp_id)
    return curr.lastrowid


def _example():
    """
    """
    conn = connect(db)
    new_experiment(conn, "qute majo", "mega kink")
    create_run(conn, "1", "sweep", "asd", "asd")
    finish_experiment(conn, "majorana_qubit")