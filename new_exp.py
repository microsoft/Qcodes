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
from contextlib import contextmanager
import logging
import sqlite3
import time
from numbers import Number
from numpy import Array
import numpy as np
import io
from typing import Any, List, Optional, Tuple, Union, Dict

db = "/Users/unga/Desktop/experiment.db"


def adapt_array(arr: ndarray)->sqlite3.Binary:
    """
    See this:
    https://stackoverflow.com/questions/3425320/sqlite3-programmingerror-you-must-not-use-8-bit-bytestrings-unless-you-use-a-te
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text: bytes)->ndarray:
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


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


def connect(name: str, debug: bool=False) -> sqlite3.Connection:
    # register numpy->binary(TEXT) adapter
    sqlite3.register_adapter(np.ndarray, adapt_array)
    # register binary(TEXT) -> numpy converter
    sqlite3.register_converter("array", convert_array)
    conn = sqlite3.connect(db, detect_types=sqlite3.PARSE_DECLTYPES)
    # sqlite3 options
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
    try:
        c = transaction(conn, sql, *args)
    except Exception as e:
        logging.exception("Could not execute transaction, rolling back")
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
                   format_string: Optional[str] = "{}-{}-{}"
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
    formatted_name = format_string.format(name, exp_id, run_counter)
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


# this is what we accept
PARAMETERS = None # shold be a list of pyton object that have .type that
# represnt the type of sqlite coumn
VALUES = List[Union[str, Number, List, Array, bool]]
# TODO: do same for metadata maybe?


class ParamSpec():
    def __init__(self, name: str, type: str, **metadata) -> None:
        self.name = name
        self.type = type
        if metadata:
            self.metadata = metadata

    def sql_repr(self):
        return f"{self.name, self.type}"


def create_run_table(conn: sqlite3.Connection,
                     formatted_name: str,
                     parameters: Optional[List[ParamSpec]]=None,
                     values: Optional[VALUES]=None,
                     metadata: Dict[str, Any]=None
                     )->None:
    """Create run table with formatted_name as name

    Args:
        conn: database connection
        formatted_name: the name of the table to create
    """
    _parameters = ",".join([p.sql_repr() for p in parameters])
    if parameters and values:
        # TODO: no need to create the table fisrt and then insert the values
        pass
    elif parameters:
        query = f"""
        CREATE TABLE "{formatted_name}" (
            id INTEGER PRIMARY KEY,
            {_parameters}
        );
        """
        transaction(conn, query)
    else:
        # look ma no parameters
        # TODO: does this even make sense?
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
    with atomic(conn):
        run_counter, formatted_name, row_id = insert_run(conn,
                                                         exp_id,
                                                         name)
        update_eperiment_run_counter(conn, exp_id, run_counter)
        create_run_table(conn, formatted_name)
    return row_id, formatted_name


def _example():
    """
    """
    conn = connect(db)
    new_experiment(conn, "qute majo", "mega kink")
    create_run(conn, "1", "sweep", "asd", "asd")
    finish_experiment(conn, "majorana_qubit")


@contextmanager
def atomic(conn: sqlite3.Connection):
    """
    Guard a series of transactions as atomic.
    If one fails the transction is rolled back and no more transactions
    are performed.

    Args:
        - conn: connection to
    """
    try:
        yield
    except Exception as e:
        conn.rollback()
        raise RuntimeError("Rolling back due to unhandled exceptio") from e
    else:
        conn.commit()
