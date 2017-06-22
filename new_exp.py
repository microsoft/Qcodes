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
from typing import Any, Optional, List, Tuple


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


def transaction(conn: sqlite3.Connection,
                sql: str, *args: Any)->sqlite3.Cursor:
    c = conn.cursor()
    if len(args) > 0:
        c.execute(sql, args)
    else:
        c.execute(sql)
    return c


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
        raise e

    conn.commit()
    return c


def insert_column(conn: sqlite3.Connection, table: str, name: str,
                  type: str)->None:
    atomicTransaction(conn,
                      f"ALTER TABLE {table} ADD COLUMN {name} {type}")


def new_experiment(conn: sqlite3.Connection,
                   name: str,
                   sample_name: str,
                   format_string: Optional[str] = "{}-{}"
                   )->int:
    """ Add new experiment to container

    Args:
        conn: database connection
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


def insert_run(conn: sqlite3.Connection, exp_id: int, name: str):
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
    curr = transaction(conn, query,
                       name,
                       exp_id,
                       formatted_name,
                       run_counter,
                       time.time()
                       )
    return run_counter, formatted_name, curr.lastrowid


def update_eperiment_run_counter(conn: sqlite3.Connection, exp_id: int,
                                 run_counter: int)->None:
    """ Update experiment with
    """
    query = """
    UPDATE experiments
    SET run_counter = ?
    WHERE exp_id = ?
    """
    transaction(conn, query, run_counter, exp_id)


def create_run_table(conn: sqlite3.Connection, formatted_name: str)->None:
    """Create run table with formatted_name as name

    Args:
        conn: database connection
        formatted_name: the name of the table to create
    """
    query = f"""
    CREATE TABLE "{formatted_name}" (
        id INTEGER PRIMARY KEY
    );
    """
    transaction(conn, query)


def create_run(conn: sqlite3.Connection, exp_id: int, name: str,
               *parameters, **metadata)-> Tuple[int, str]:
    """ Create a single run for the experiment.


    This will register the run in the runs table, the counter in the
    experiments table and create a new table with the formatted name.
    The operations are NOT atomic, but the function is.
    NOTE: this function is not idempotent.

    Args:
        - conn: the connection to the sqlite database
        - exp_id: the experiment id we want to create the run into
        - name: a friendly name for this run
        - paramters : TODO:
        - metadata : TODO:

    Returns:
        - run_id: the row id of the newly created run
        - formatted_name: the name of the newly created table
    """
    try:
        run_counter, formatted_name, row_id = insert_run(conn,
                                                         exp_id,
                                                         name)
        update_eperiment_run_counter(conn, exp_id, run_counter)
        create_run_table(conn, formatted_name)
    except Exception as e:
        conn.rollback
        raise e
    conn.commit()
    return row_id, formatted_name


def _example():
    """
    """
    conn = connect(db)
    new_experiment(conn, "qute majo", "mega kink")
    create_run(conn, "1", "sweep", "asd", "asd")
    finish_experiment(conn, "majorana_qubit")
