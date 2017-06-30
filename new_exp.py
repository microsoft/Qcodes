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
from numpy import ndarray
import numpy as np
import io
import socket
from typing import Any, List, Optional, Tuple, Union, Dict


# TODO: need to do run init.sql somewhere somehow
# maybe move the sql script into a giant string here

# TODO: this clearly should be configurable
db = "/Users/unga/Desktop/experiment.db"


# represent the type of  data we can/want map to sqlite column
VALUES = List[Union[str, Number, List, ndarray, bool]]


class ParamSpec():
    def __init__(self, name: str, type: str, **metadata) -> None:
        self.name = name
        self.type = type
        if metadata:
            self.metadata = metadata

    def sql_repr(self):
        return f"{self.name} {self.type}"

    def __repr__(self):
        return f"{self.name} ({self.type})"


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
    # for some reasons mypy complains about this
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


@contextmanager
def atomic(conn: sqlite3.Connection):
    """
    Guard a series of transactions as atomic.
    If one fails the transaction is rolled back and no more transactions
    are performed.

    Args:
        - conn: connection to guard
    """
    try:
        yield
    except Exception as e:
        conn.rollback()
        raise RuntimeError("Rolling back due to unhandled exception") from e
    else:
        conn.commit()


def insert_column(conn: sqlite3.Connection, table: str, name: str,
                  type: Optional[str]=None)->None:
    if type:
        transaction(conn,
                    f'ALTER TABLE "{table}" ADD COLUMN "{name}" {type}')
    else:
        transaction(conn,
                    f'ALTER TABLE "{table}" ADD COLUMN "{name}"')


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
    Returns:
        id: row-id of the created experiment
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
    """ Finish experiment

    Args:
        conn: database
        name: the name of the experiment
    """
    query = """
    UPDATE experiments SET end_time=? WHERE exp_id=?;
    """
    atomicTransaction(conn, query, time.time(), exp_id)


def data_sets(conn: sqlite3.Connection):
    sql = """
    SELECT * from runs
    """
    c = transaction(conn, sql)
    return c.fetchall()


def experiments(conn: sqlite3.Connection):
    sql = """
    SELECT * from experiments
    """
    c = transaction(conn, sql)
    return c.fetchall()


def _select_one_where(conn: sqlite3.Connection, table: str, column: str,
                      where_column: str, where_value: Any) -> Any:
    query = f"""
    SELECT {column}
    FROM
        {table}
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
        {table}
    WHERE
        {where_column} = ?
    """
    cur = atomicTransaction(conn, query, where_value)
    res = many(cur, *columns)
    return res


# TODO: can make many of those. Easier to use // enforce some types
# but slower because make one query only
def get_run_counter(conn: sqlite3.Connection, exp_id: int) -> int:
    return _select_one_where(conn, "experiments", "run_counter",
                             where_column="exp_id",
                             where_value=exp_id)


def insert_meta_data(conn: sqlite3.Connection, run_id: int,
                     metadata: Dict[str, Any])->None:
    """
    Creates new medata data (they must exist already)
    """
    for key, value in metadata.items():
        insert_column(conn, "runs", key)
        sql = f"""
            UPDATE runs set '{key}'=? WHERE rowid=?;
        """
        transaction(conn, sql, value, run_id)


def _massage_medata(metadata: Dict[str, Any])-> Tuple[str, List[Any]]:
    template = []
    values = []
    for key, value in metadata.items():
        template.append(f"{key} = ?")
        values.append(value)
    return ','.join(template), values


def update_meta_data(conn: sqlite3.Connection, run_id: int,
                     metadata: Dict[str, Any])->None:
    """
    Updates medata data (they must exist already)
    """
    template, values = _massage_medata(metadata)
    sql = f"""
    UPDATE runs set
        {template}
    WHERE rowid=?;
    """
    transaction(conn, sql, *values, run_id)


def add_meta_data(conn: sqlite3.Connection, run_id: int,
                  metadata: Dict[str, Any])->None:
    """
    Add medata data (updates if exists, create otherwise)
    """
    with atomic(conn):
        try:
            insert_meta_data(conn, run_id, metadata)
        except sqlite3.OperationalError as e:
            # this means that the column already exists
            # so just insert the new value
            if str(e).startswith("duplicate"):
        else:
            raise e


def insert_run(conn: sqlite3.Connection, exp_id: int, name: str,
               parameters: Optional[List[ParamSpec]]=None,
               ):
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
    if parameters:
        query = f"""
        INSERT INTO {table}
            (name,exp_id,result_table_name,result_counter,run_timestamp,parameters)
        VALUES
            (?,?,?,?,?,?)
        """
        curr = transaction(conn, query,
                           name,
                           exp_id,
                           formatted_name,
                           run_counter,
                           time.time(),
                           ",".join([p.name for p in parameters])
                           )
    else:
        query = f"""
        INSERT INTO {table}
            (name,exp_id,result_table_name,result_counter,run_timestamp)
        VALUES
            (?,?,?,?,?,?)
        """
        curr = transaction(conn, query,
                           name,
                           exp_id,
                           formatted_name,
                           run_counter,
                           time.time()
                           )
    return run_counter, formatted_name, curr.lastrowid


def update_experiment_run_counter(conn: sqlite3.Connection, exp_id: int,
                                  run_counter: int)->None:
    """ Update experiment with
    """
    query = """
    UPDATE experiments
    SET run_counter = ?
    WHERE exp_id = ?
    """
    transaction(conn, query, run_counter, exp_id)


def insert_values(conn: sqlite3.Connection,
                  formatted_name: str,
                  parameters: List[ParamSpec],
                  values: VALUES,
                  )->int:
    """
    Inserts values for the corresponding paramSpec
    Will pad with null if not all parameters are specified.
    NOTE this need to be committed before closing the connection.
    """
    _parameters = ",".join([p.name for p in parameters])
    _values = ",".join(["?"]*len(parameters))
    query = f"""INSERT INTO "{formatted_name}"
        ({_parameters})
    VALUES
        ({_values})
    """
    # this will raise an error if there is a mismatch
    # between the values and parameters length
    # TODO: check inputs instead?
    c = transaction(conn, query, *values)
    return c.lastrowid


def insert_many_values(conn: sqlite3.Connection,
                       formatted_name: str,
                       parameters: List[ParamSpec],
                       values: List[VALUES],
                       )->int:
    """
    Inserts many values for the corresponding paramSpec.
    Will pad with null if not all parameters are specified.

    NOTE this need to be committed before closing the connection.
    """
    _parameters = ",".join([p.name for p in parameters])
    # TODO: none of the code below is not form PRADA SS-2017
    # [a, b] -> (?,?), (?,?)
    # [[1,1], [2,2]]
    _values = "("+",".join(["?"]*len(parameters))+")"
    # NOTE: assume that all the values have same length
    _values_x_params = ",".join([_values]*len(values[0]))
    query = f"""INSERT INTO "{formatted_name}"
        ({_parameters})
    VALUES
        {_values_x_params}
    """
    # this will raise an error if there is a mismatch
    # between the values and parameters length
    # TODO: check inputs instead?
    # we need to make values a flat list from a list of list
    flattened_values = [item for sublist in values for item in sublist]
    c = transaction(conn, query, *flattened_values)
    return c.lastrowid


def modify_values(conn: sqlite3.Connection,
                  formatted_name: str,
                  index: int,
                  parameters: List[ParamSpec],
                  values: VALUES,
                  )->int:
    """
    Modify values for the corresponding paramSpec
    If a parameter is in the table but not in the parameter list is
    left untouched.
    If a parameter is mapped to None, it will be a null value.
    """
    name_val_template = []
    for name, value in zip([p.name for p in parameters], values):
        name_val_template.append(f"{name}=?")
    name_val_templates = ",".join(name_val_template)
    query = f"""
    UPDATE "{formatted_name}"
    SET
        {name_val_templates}
    WHERE
        rowid = {index+1}
    """
    # this will raise an error if there is a mismatch
    # between the values and parameters length
    # TODO: check inputs instead?
    c = atomicTransaction(conn, query, *values)
    return c.rowcount


def length(conn: sqlite3.Connection,
           formatted_name: str
           )-> int:
    """
    Return the lenght of the table

    Args:
        - conn: the connection to the sqlite database
        - formatted_name: name of the table

    Returns:
        -the lenght of the table
    """
    query = f"select MAX(id) from '{formatted_name}'"
    c = atomicTransaction(conn, query)
    return c.fetchall()[0][0]


def last_experiment(conn: sqlite3.Connection) -> int:
    """
    Return last started experiment id
    """
    query = "select MAX(exp_id) from experiments"
    c = atomicTransaction(conn, query)
    return c.fetchall()[0][0]


def modify_many_values(conn: sqlite3.Connection,
                       formatted_name: str,
                       start_index: int,
                       parameters: List[ParamSpec],
                       values: List[VALUES],
                       )->None:
    """
    Modify many values for the corresponding paramSpec
    If a parameter is in the table but not in the parameter list is
    left untouched.
    If a parameter is mapped to None, it will be a null value.
    """
    _len = length(conn, formatted_name)
    len_requested = start_index + len(values)
    available = _len - start_index
    if len_requested > _len:
        reason = f""""Modify operation Out of bounds.
        Trying to modify {len(values)} results,
        but therere are only {available} retulst.
        """
        raise ValueError(reason)
    for value in values:
        modify_values(conn, formatted_name, start_index, parameters, value)
        start_index += 1


def get_parameters(conn: sqlite3.Connection,
                   formatted_name: str) -> List[ParamSpec]:
    """
    gets the list of param specs for run

    Args:
        - conn: the connection to the sqlite database
        - formatted_name: name of the table

    Returns:
        - A list of param specs for this table
    """
    # TODO: FIX mapping of types
    c = conn.execute(f"""pragma table_info('{formatted_name}')""")
    params: List[ParamSpec] = []
    for row in c.fetchall():
        if row['name'] == 'id':
            continue
        else:
            params.append(ParamSpec(row['name'], row['type']))
    return params


def add_parameter(conn: sqlite3.Connection,
                  formatted_name: str,
                  *parameter: ParamSpec):
    with atomic(conn):
        for p in parameter:
            insert_column(conn, formatted_name, p.name, p.type)


def get_last_run(conn: sqlite3.Connection, exp_id: int) -> str:
    query = """
    SELECT result_table_name, max(run_timestamp), exp_id
    FROM runs
    WHERe exp_id = ?;
    """
    c = transaction(conn, query, exp_id)
    return one(c, 'result_table_name')


def create_run_table(conn: sqlite3.Connection,
                     formatted_name: str,
                     parameters: Optional[List[ParamSpec]]=None,
                     values: Optional[VALUES]=None
                     )->None:
    """Create run table with formatted_name as name

    NOTE this need to be committed before closing the connection.

    Args:
        conn: database connection
        formatted_name: the name of the table to create
    """
    if parameters and values:
        _parameters = ",".join([p.sql_repr() for p in parameters])
        query = f"""
        CREATE TABLE "{formatted_name}" (
            id INTEGER PRIMARY KEY,
            {_parameters}
        );
        """
        transaction(conn, query)
        # now insert values
        insert_values(conn, formatted_name, parameters, values)
    elif parameters:
        _parameters = ",".join([p.sql_repr() for p in parameters])
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


def remove_run(conn: sqlite3.Connection, formatted_name: str)->None:
    """ Delete run from experiment

    Args:
        - conn: the connection to the sqlite database
        - formatted_name: the name of the run to remove
    """
    sql = f"""
    DROP TABLE [IF EXISTS] {formatted_name}
    """
    atomicTransaction(conn, sql)


def create_run(conn: sqlite3.Connection, exp_id: int, name: str,
               *parameters: ParamSpec,
               metadata: Optional[Dict[str, Any]]=None)-> Tuple[int, str]:
    """ Create a single run for the experiment.


    This will register the run in the runs table, the counter in the
    experiments table and create a new table with the formatted name.
    The operations are NOT atomic, but the function is.
    NOTE: this function is not idempotent.

    Args:
        - conn: the connection to the sqlite database
        - exp_id: the experiment id we want to create the run into
        - name: a friendly name for this run
        - parameters : TODO:
        - metadata : TODO:

    Returns:
        - run_id: the row id of the newly created run
        - formatted_name: the name of the newly created table
    """
    with atomic(conn):
        run_counter, formatted_name, run_id = insert_run(conn,
                                                         exp_id,
                                                         name,
                                                         list(parameters))
        add_meta_data(conn, run_id, metadata)
        update_experiment_run_counter(conn, exp_id, run_counter)
        # NOTE: cast to list to make mypy happy (my bug, mypy bug?)
        create_run_table(conn, formatted_name, list(parameters))
    return run_id, formatted_name


def get_data(conn: sqlite3.Connection,
             formatted_name: str,
             parameters: List[ParamSpec],
             start: int=None,
             end: int=None,
             )->Any:
    _parameters = ",".join([p.name for p in parameters])
    if start and end:
        query = f"""
        SELECT {_parameters}
        FROM "{formatted_name}"
        WHERE rowid
            > {start} and
              rowid
            <= {end}
        """
    elif start:
        query = f"""
        SELECT {_parameters}
        FROM "{formatted_name}"
        WHERE rowid
            >= {start}
        """
    elif end:
        query = f"""
        SELECT {_parameters}
        FROM "{formatted_name}"
        WHERE rowid
            <= {end}
        """
    else:
        query = f"""
        SELECT {_parameters}
        FROM "{formatted_name}"
        """
    c = transaction(conn, query)
    res = many(c, *[p.name for p in parameters])
    return res



