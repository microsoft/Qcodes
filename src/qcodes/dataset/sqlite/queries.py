"""
This module contains useful SQL queries and their combinations which are
specific to the domain of QCoDeS database.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import unicodedata
import warnings
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import TypedDict

import qcodes as qc
from qcodes import config
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning import serialization as serial
from qcodes.dataset.descriptions.versioning import v0
from qcodes.dataset.descriptions.versioning.converters import new_to_old, old_to_new
from qcodes.dataset.guids import build_guid_from_components, parse_guid
from qcodes.dataset.sqlite.connection import (
    AtomicConnection,
    atomic,
    atomic_transaction,
    transaction,
)
from qcodes.dataset.sqlite.query_helpers import (
    VALUE,
    VALUES,
    get_description_map,
    insert_column,
    insert_values,
    is_column_in_table,
    many,
    many_many,
    one,
    select_many_where,
    select_one_where,
    sql_placeholder_string,
    update_where,
)
from qcodes.utils import list_of_data_to_maybe_ragged_nd_array

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

log = logging.getLogger(__name__)


_unicode_categories: tuple[
    Literal["Lu"],
    Literal["Ll"],
    Literal["Lt"],
    Literal["Lm"],
    Literal["Lo"],
    Literal["Nd"],
    Literal["Pc"],
    Literal["Pd"],
    Literal["Zs"],
] = ("Lu", "Ll", "Lt", "Lm", "Lo", "Nd", "Pc", "Pd", "Zs")


# in the current version, these are the standard columns of the "runs" table
# Everything else is metadata
RUNS_TABLE_COLUMNS = (
    "run_id",
    "exp_id",
    "name",
    "result_table_name",
    "result_counter",
    "run_timestamp",
    "completed_timestamp",
    "is_completed",
    "parameters",
    "guid",
    "run_description",
    "snapshot",
    "parent_datasets",
    "captured_run_id",
    "captured_counter",
)


def is_run_id_in_database(conn: AtomicConnection, *run_ids: int) -> dict[int, bool]:
    """
    Look up run_ids and return a dictionary with the answers to the question
    "is this run_id in the database?"

    Args:
        conn: the connection to the database
        run_ids: the run_ids to look up

    Returns:
        a dict with the run_ids as keys and bools as values. True means that
        the run_id DOES exist in the database

    """
    run_ids = tuple(np.unique(run_ids))
    placeholders = sql_placeholder_string(len(run_ids))

    query = f"""
             SELECT run_id
             FROM runs
             WHERE run_id in {placeholders}
            """

    cursor = conn.cursor()
    cursor.execute(query, run_ids)
    rows = cursor.fetchall()
    existing_ids = [row[0] for row in rows]
    return {run_id: (run_id in existing_ids) for run_id in run_ids}


def get_parameter_data(
    conn: AtomicConnection,
    table_name: str,
    columns: Sequence[str] = (),
    start: int | None = None,
    end: int | None = None,
    callback: Callable[[float], None] | None = None,
) -> dict[str, dict[str, npt.NDArray]]:
    """
    Get data for one or more parameters and its dependencies. The data
    is returned as numpy arrays within 2 layers of nested dicts. The keys of
    the outermost dict are the requested parameters and the keys of the second
    level are the loaded parameters (requested parameter followed by its
    dependencies). Start and End allows one to specify a range of rows to
    be returned (1-based indexing, both ends are included). The range filter
    is applied AFTER the NULL values have been filtered out.
    Be aware that different parameters that are independent of each other
    may return a different number of rows.

    Note that this assumes that all array type parameters have the same length.
    This should always be the case for a parameter and its dependencies.

    Note that all numeric data will at the moment be returned as floating point
    values.

    Args:
        conn: database connection
        table_name: name of the table
        columns: list of columns. If no columns are provided, all parameters
            are returned.
        start: start of range; if None, then starts from the top of the table
        end: end of range; if None, then ends at the bottom of the table
        callback: Function called during the data loading every
            config.dataset.callback_percent.

    """
    rundescriber = get_rundescriber_from_result_table_name(conn, table_name)

    output = {}
    if len(columns) == 0:
        columns = [ps.name for ps in rundescriber.interdeps.top_level_parameters]

    # loop over all the requested parameters
    for output_param in columns:
        output[output_param] = get_shaped_parameter_data_for_one_paramtree(
            conn, table_name, rundescriber, output_param, start, end, callback
        )
    return output


def get_shaped_parameter_data_for_one_paramtree(
    conn: AtomicConnection,
    table_name: str,
    rundescriber: RunDescriber,
    output_param: str,
    start: int | None,
    end: int | None,
    callback: Callable[[float], None] | None = None,
) -> dict[str, npt.NDArray]:
    """
    Get the data for a parameter tree and reshape it according to the
    metadata about the dataset. This will only reshape the loaded data if
    the number of points in the loaded data matches the expected number of
    points registered in the metadata.
    If there are more measured datapoints
    than expected a warning will be given.
    """

    one_param_output, _ = get_parameter_data_for_one_paramtree(
        conn, table_name, rundescriber, output_param, start, end, callback
    )
    if rundescriber.shapes is not None:
        shape = rundescriber.shapes.get(output_param)

        if shape is not None:
            # if the user calculates shape its very easy to end up putting in a
            # float such as 100.0 rather than 100 so to be safe we cast to int here
            shape = tuple(int(s) for s in shape)
            total_len_shape = np.prod(shape)
            for name, paramdata in one_param_output.items():
                total_data_shape = np.prod(paramdata.shape)
                if total_data_shape == total_len_shape:
                    one_param_output[name] = paramdata.reshape(shape)
                elif total_data_shape > total_len_shape:
                    log.warning(
                        f"Tried to set data shape for {name} in "
                        f"dataset {output_param} "
                        f"from metadata when "
                        f"loading but found inconsistent lengths "
                        f"{total_data_shape} and {total_len_shape}"
                    )
    return one_param_output


def get_rundescriber_from_result_table_name(
    conn: AtomicConnection, result_table_name: str
) -> RunDescriber:
    sql = """
    SELECT run_id FROM runs WHERE result_table_name = ?
    """
    c = atomic_transaction(conn, sql, result_table_name)
    run_id = one(c, "run_id")
    rd = serial.from_json_to_current(get_run_description(conn, run_id))
    return rd


def get_parameter_data_for_one_paramtree(
    conn: AtomicConnection,
    table_name: str,
    rundescriber: RunDescriber,
    output_param: str,
    start: int | None,
    end: int | None,
    callback: Callable[[float], None] | None = None,
) -> tuple[dict[str, npt.NDArray], int]:
    interdeps = rundescriber.interdeps
    data, paramspecs, n_rows = _get_data_for_one_param_tree(
        conn, table_name, interdeps, output_param, start, end, callback
    )
    if not paramspecs[0].name == output_param:
        raise ValueError(
            "output_param should always be the first "
            "parameter in a parameter tree. It is not"
        )
    _expand_data_to_arrays(data, paramspecs)

    param_data = {}
    # Benchmarking shows that transposing the data with python types is
    # faster than transposing the data using np.array.transpose
    res_t = map(list, zip(*data))

    specs_and_data = zip_longest(paramspecs, res_t, fillvalue=())

    for paramspec, column_data in specs_and_data:
        assert isinstance(paramspec, ParamSpecBase)
        # its not obvious how to type that paramspecs is always
        # longer than res_t
        if paramspec.type == "numeric":
            # there is no reliable way to
            # tell the difference between a float and and int loaded
            # from sqlite numeric columns so always fall back to float
            dtype: type | None = np.float64
        else:
            dtype = None
        param_data[paramspec.name] = list_of_data_to_maybe_ragged_nd_array(
            column_data, dtype
        )
    return param_data, n_rows


def _expand_data_to_arrays(
    data: list[tuple[Any, ...]], paramspecs: Sequence[ParamSpecBase]
) -> None:
    types = [param.type for param in paramspecs]
    # if we have array type parameters expand all other parameters
    # to arrays
    if "array" in types:
        if "numeric" in types or "text" in types or "complex" in types:
            first_array_element = types.index("array")
            types_mapping: dict[int, Callable[[str], np.dtype[Any]]] = {}
            for i, x in enumerate(types):
                if x == "numeric":
                    types_mapping[i] = lambda _: np.dtype(np.float64)
                elif x == "complex":
                    types_mapping[i] = lambda _: np.dtype(np.complex128)
                elif x == "text":
                    types_mapping[i] = lambda array: np.dtype(f"U{len(array)}")

            for i_row, row in enumerate(data):
                # todo should we handle int/float types here
                # we would in practice have to perform another
                # loop to check that all elements of a given can be cast to
                # int without loosing precision before choosing an integer
                # representation of the array
                data[i_row] = tuple(
                    np.full_like(
                        row[first_array_element], array, dtype=types_mapping[i](array)
                    )
                    if i in types_mapping
                    else array
                    for i, array in enumerate(row)
                )

        row_shape = None
        for i_row, row in enumerate(data):
            # now expand all one element arrays to match the expected size
            # one element arrays are introduced if scalar values are stored
            # with an explicit array storage type
            max_size = 0
            for i, array in enumerate(row):
                if array.size > max_size:
                    if max_size > 1:
                        log.warning(
                            f"Cannot expand array of size {max_size} "
                            f"to size {array.size}"
                        )
                    max_size, row_shape = array.size, array.shape

            if max_size > 1:
                assert row_shape is not None
                data[i_row] = tuple(
                    np.full(row_shape, array, dtype=array.dtype)
                    if array.size == 1
                    else array
                    for array in row
                )


def _get_data_for_one_param_tree(
    conn: AtomicConnection,
    table_name: str,
    interdeps: InterDependencies_,
    output_param: str,
    start: int | None,
    end: int | None,
    callback: Callable[[float], None] | None = None,
) -> tuple[list[tuple[Any, ...]], tuple[ParamSpecBase, ...], int]:
    output_param_spec = interdeps._id_to_paramspec[output_param]
    # find all the dependencies of this param

    top_level_param, deps, infs = interdeps.all_parameters_in_tree_by_group(
        output_param_spec
    )
    dep_and_infs = list(deps) + list(infs)
    dep_and_inf_names = [param.name for param in dep_and_infs]
    res = get_parameter_tree_values(
        conn,
        table_name,
        output_param,
        *dep_and_inf_names,
        start=start,
        end=end,
        callback=callback,
    )
    n_rows = len(res)
    return res, (top_level_param, *dep_and_infs), n_rows


def get_parameter_db_row(
    conn: AtomicConnection, table_name: str, param_name: str
) -> int:
    """
    Get the total number of not-null values of a parameter

    Args:
        conn: Connection to the database
        table_name: Name of the table that holds the data
        param_name: Name of the parameter to get the setpoints of

    Returns:
        The total number of not-null values

    """
    sql = f"""
           SELECT COUNT({param_name}) FROM "{table_name}"
           WHERE {param_name} IS NOT NULL
           """
    c = atomic_transaction(conn, sql)

    return one(c, 0)


def get_table_max_id(conn: AtomicConnection, table_name: str) -> int:
    """
    Get the max id of a table

    Args:
        conn: Connection to the database
        table_name: Name of the table that holds the data

    Returns:
        The max id of a table

    """
    sql = f"""
           SELECT MAX(id)
           FROM "{table_name}"
           """
    c = atomic_transaction(conn, sql)

    return one(c, 0)


def _get_offset_limit_for_callback(
    conn: AtomicConnection, table_name: str, param_name: str
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Since sqlite3 does not allow to keep track of the data loading progress,
    we compute how many sqlite request correspond to a progress of
    config.dataset.callback_percent.
    This function return a list of offset and a integer value of limit to
    be used to run such SQL requests.

    Args:
        conn: Connection to the database
        table_name: Name of the table that holds the data
        param_name: Name of the parameter to get the setpoints of

    Returns:
        offset: list of SQL offset corresponding to a progress of
            config.dataset.callback_percent
        limit: SQL limit corresponding to a progress of
            config.dataset.callback_percent

    """

    # First, we get the number of row to be downloaded for the wanted
    # dependent parameter
    nb_row = get_parameter_db_row(conn, table_name, param_name)

    # Second, we get the max id of the table
    max_id = get_table_max_id(conn, table_name)

    # Third, we create a list of offset corresponding to a progress of
    # config.dataset.callback_percent
    if nb_row >= 100:
        # Using linspace with dtype=int ensure of having an array finishing
        # by max_id
        offset: npt.NDArray[np.int32] = np.linspace(
            0, max_id, int(100 / config.dataset.callback_percent) + 1, dtype=int
        )

    else:
        # If there is less than 100 row to be downloaded, we overwrite the
        # config.dataset.callback_percent to avoid many calls for small download
        offset = np.array([0, nb_row // 2, nb_row])

    # The number of row downloaded between two iterations may vary
    # We compute the limit corresponding to each offset
    limit = offset[1:] - offset[:-1]

    return offset, limit


def get_parameter_tree_values(
    conn: AtomicConnection,
    result_table_name: str,
    toplevel_param_name: str,
    *other_param_names: str,
    start: int | None = None,
    end: int | None = None,
    callback: Callable[[float], None] | None = None,
) -> list[tuple[Any, ...]]:
    """
    Get the values of one or more columns from a data table. The rows
    retrieved are the rows where the 'toplevel_param_name' column has
    non-NULL values, which is useful when retrieving a top level parameter
    and its setpoints (and inferred_from parameter values)

    Args:
        conn: Connection to the DB file
        result_table_name: The result table whence the values are to be
            retrieved
        toplevel_param_name: Name of the column that holds the top level
            parameter
        other_param_names: Names of additional columns to retrieve
        start: The (1-indexed) result to include as the first results to
            be returned. None is equivalent to 1. If start > end, nothing
            is returned.
        end: The (1-indexed) result to include as the last result to be
            returned. None is equivalent to "all the rest". If start > end,
            nothing is returned.
        callback: Function called during the data loading every
            config.dataset.callback_percent.

    Returns:
        A list of list. The outer list index is row number, the inner list
        index is parameter value (first toplevel_param, then other_param_names)

    """

    cursor = conn.cursor()

    # Without callback: int
    # With callback: npt.NDArray
    offset: int | npt.NDArray
    limit: int | npt.NDArray

    offset = max((start - 1), 0) if start is not None else 0
    limit = max((end - offset), 0) if end is not None else -1

    if start is not None and end is not None and start > end:
        limit = 0

    # start and end currently not working with callback
    if start is None and end is None and callback is not None:
        offset, limit = _get_offset_limit_for_callback(
            conn, result_table_name, toplevel_param_name
        )

    # Create the base sql query
    columns = [toplevel_param_name, *list(other_param_names)]
    sql = f"""
           SELECT "{'","'.join(columns)}" FROM "{result_table_name}"
           WHERE {toplevel_param_name} IS NOT NULL
           LIMIT ? OFFSET ?
           """

    # Request if no callback
    if callback is None:
        cursor.execute(sql, (limit, offset))
        res = many_many(cursor, *columns)

    # Request if callback
    elif callback is not None:
        assert isinstance(offset, np.ndarray)
        assert isinstance(limit, np.ndarray)
        progress_current = 100 / len(limit)

        # 0
        progress_total = 0.0
        callback(progress_total)

        # 1
        cursor.execute(sql, (limit[0], offset[0]))
        res = many_many(cursor, *columns)
        progress_total += progress_current
        callback(progress_total)

        # others
        for i in range(1, len(offset) - 1):
            cursor.execute(sql, (limit[i], offset[i]))
            res.extend(many_many(cursor, *columns))
            progress_total += progress_current
            callback(progress_total)

    return res


def get_runid_from_expid_and_counter(
    conn: AtomicConnection, exp_id: int, counter: int
) -> int:
    """
    Get the run_id of a run in the specified experiment with the specified
    counter

    Args:
        conn: connection to the database
        exp_id: the exp_id of the experiment containing the run
        counter: the intra-experiment run counter of that run

    """
    sql = """
          SELECT run_id
          FROM runs
          WHERE result_counter= ? AND
          exp_id = ?
          """
    c = transaction(conn, sql, counter, exp_id)
    run_id = one(c, "run_id")
    return run_id


def get_guid_from_expid_and_counter(
    conn: AtomicConnection, exp_id: int, counter: int
) -> str:
    """
    Get the guid of a run in the specified experiment with the specified
    counter

    Args:
        conn: connection to the database
        exp_id: the exp_id of the experiment containing the run
        counter: the intra-experiment run counter of that run

    """
    sql = """
          SELECT guid
          FROM runs
          WHERE result_counter= ? AND
          exp_id = ?
          """
    c = transaction(conn, sql, counter, exp_id)
    run_id = one(c, "guid")
    return run_id


def get_runid_from_guid(conn: AtomicConnection, guid: str) -> int | None:
    """
    Get the run_id of a run based on the guid

    Args:
        conn: connection to the database
        guid: the guid to look up

    Returns:
        The run_id if found, else None

    Raises:
        RuntimeError if more than one run with the given GUID exists

    """
    query = """
            SELECT run_id
            FROM runs
            WHERE guid = ?
            """
    cursor = conn.cursor()
    cursor.execute(query, (guid,))
    rows = cursor.fetchall()
    if len(rows) == 0:
        run_id = None
    elif len(rows) > 1:
        errormssg = (
            "Critical consistency error: multiple runs with"
            f" the same GUID found! {len(rows)} runs have GUID "
            f"{guid}"
        )
        log.critical(errormssg)
        raise RuntimeError(errormssg)
    else:
        run_id = int(rows[0][0])

    return run_id


def _query_guids_from_run_spec(
    conn: AtomicConnection,
    captured_run_id: int | None = None,
    captured_counter: int | None = None,
    experiment_name: str | None = None,
    sample_name: str | None = None,
) -> list[str]:
    """
    Get the GUIDs of runs matching the supplied run specifications.

    # Todo: do we need to select by start/end time too? Is result name useful?

    Args:
        conn: connection to the database.
        captured_run_id: the run_id that was assigned to this
            run at capture time.
        captured_counter: the counter that was assigned to this
            run at capture time.
        experiment_name: Name of the experiment that the runs should belong to.
        sample_name: Name of the sample that the query should be restricted to.

    Returns:
        A list of the GUIDs matching the supplied specifications.

    """
    # first find all experiments that match the given sample
    # and experiment name
    exp_query = {}
    exp_ids: list[int] | None
    if experiment_name is not None or sample_name is not None:
        if sample_name is not None:
            exp_query["sample_name"] = sample_name
        if experiment_name is not None:
            exp_query["name"] = experiment_name
        exp_ids = get_matching_exp_ids(conn, **exp_query)
        if exp_ids == []:
            return []
    else:
        exp_ids = None

    conds = []
    inputs = []

    if exp_ids is not None:
        exp_placeholder = sql_placeholder_string(len(exp_ids))
        conds.append(f"exp_id in {exp_placeholder}")
        inputs.extend(exp_ids)
    if captured_run_id is not None:
        conds.append("captured_run_id is ?")
        inputs.append(captured_run_id)
    if captured_counter is not None:
        conds.append("captured_counter is ?")
        inputs.append(captured_counter)

    if len(conds) >= 1:
        where_clause = " WHERE " + " AND ".join(conds)
    else:
        where_clause = ""

    query = "SELECT guid from runs" + where_clause + " ORDER BY run_id"

    cursor = conn.cursor()
    if len(inputs) > 0:
        cursor.execute(query, inputs)
    else:
        cursor.execute(query)

    return [guid for (guid,) in cursor.fetchall()]


def _get_layout_id(
    conn: AtomicConnection, parameter: ParamSpec | str, run_id: int
) -> int:
    """
    Get the layout id of a parameter in a given run

    Args:
        conn: The database connection
        parameter: A ParamSpec or the name of the parameter
        run_id: The run_id of the run in question

    """
    # get the parameter layout id
    sql = """
    SELECT layout_id FROM layouts
    WHERE parameter = ?
    and run_id = ?
    """

    if isinstance(parameter, ParamSpec):
        name = parameter.name
    elif isinstance(parameter, str):
        name = parameter
    else:
        raise ValueError(
            "Wrong parameter type, must be ParamSpec or str, "
            f"received {type(parameter)}."
        )

    c = atomic_transaction(conn, sql, name, run_id)
    res = one(c, "layout_id")

    return res


def _get_dependents(conn: AtomicConnection, run_id: int) -> list[int]:
    """
    Get dependent layout_ids for a certain run_id, i.e. the layout_ids of all
    the dependent variables
    """
    sql = """
    SELECT layout_id FROM layouts
    WHERE run_id=? and layout_id in (SELECT dependent FROM dependencies)
    """
    c = atomic_transaction(conn, sql, run_id)
    res = [layout_id for (layout_id,) in many_many(c, "layout_id")]
    return res


def _get_dependencies(conn: AtomicConnection, layout_id: int) -> list[tuple[int, int]]:
    """
    Get the dependencies of a certain dependent variable (indexed by its
    layout_id)

    Args:
        conn: connection to the database
        layout_id: the layout_id of the dependent variable

    """
    sql = """
    SELECT independent, axis_num FROM dependencies WHERE dependent=?
    """
    c = atomic_transaction(conn, sql, layout_id)
    return c.fetchall()


# Higher level Wrappers


def new_experiment(
    conn: AtomicConnection,
    name: str,
    sample_name: str,
    format_string: str | None = "{}-{}-{}",
    start_time: float | None = None,
    end_time: float | None = None,
) -> int:
    """
    Add new experiment to container.

    Args:
        conn: database connection
        name: the name of the experiment
        sample_name: the name of the current sample
        format_string: basic format string for table-name
          must contain 3 placeholders.
        start_time: time when the experiment was started. Do not supply this
          unless you have a very good reason to do so.
        end_time: time when the experiment was completed. Do not supply this
          unless you have a VERY good reason to do so

    Returns:
        id: row-id of the created experiment

    """
    query = """
            INSERT INTO experiments
            (name, sample_name, format_string,
            run_counter, start_time, end_time)
            VALUES
            (?,?,?,?,?,?)
            """

    start_time = start_time or time.time()
    values = (name, sample_name, format_string, 0, start_time, end_time)
    curr = atomic_transaction(conn, query, *values)

    return_value = curr.lastrowid

    if return_value is None:
        raise RuntimeError(f"Insert of new experiment with {name} failed")
    return return_value


# TODO(WilliamHPNielsen): we should remove the redundant
# is_completed
def mark_run_complete(
    conn: AtomicConnection,
    run_id: int,
    timestamp: float | None = None,
    override: bool = False,
) -> None:
    """Mark run complete

    Args:
        conn: database connection
        run_id: id of the run to mark complete
        timestamp: time stamp for completion. If None the function will
            automatically get the current time.
        override: Mark already completed run complted again and override completed
            timestamp with new value. If False marking an already completed run completed
            is a noop.

    """
    if override is False:
        if completed(conn=conn, run_id=run_id):
            log.warning("Trying to mark a run completed that was already completed.")
            return

    query = """
    UPDATE
        runs
    SET
        completed_timestamp=?,
        is_completed=?
    WHERE run_id=?;
    """
    if timestamp is None:
        timestamp = time.time()
    atomic_transaction(conn, query, timestamp, True, run_id)


def completed(conn: AtomicConnection, run_id: int) -> bool:
    """Check if the run is complete

    Args:
        conn: database connection
        run_id: id of the run to check

    """
    return bool(select_one_where(conn, "runs", "is_completed", "run_id", run_id))


def get_completed_timestamp_from_run_id(
    conn: AtomicConnection, run_id: int
) -> float | None:
    """
    Retrieve the timestamp when the given measurement run was completed

    If the measurement run has not been marked as completed, then the returned
    value is None.

    Args:
        conn: database connection
        run_id: id of the run

    Returns:
        timestamp in seconds since the Epoch, or None

    """
    ts = select_one_where(conn, "runs", "completed_timestamp", "run_id", run_id)
    # sometimes it happens that the timestamp is written to DB as an int
    if isinstance(ts, int):
        ts = float(ts)
    assert isinstance(ts, (float, type(None)))
    return ts


def get_guid_from_run_id(conn: AtomicConnection, run_id: int) -> str | None:
    """
    Get the guid of the given run. Returns None if the run is not found

    Args:
        conn: database connection
        run_id: id of the run

    Returns:
        The guid of the run_id.

    """
    try:
        guid = select_one_where(conn, "runs", "guid", "run_id", run_id)
    except RuntimeError:
        return None
    assert isinstance(guid, str)
    return guid


def get_guids_from_multiple_run_ids(
    conn: AtomicConnection, run_ids: Iterable[int]
) -> list[str]:
    """
    Retrieve guids of runs in the connected database specified by their run ids.
    run ids are run_id in the database and not captured_run_id.

    Args:
        conn: database connection
        run_ids: An integer iterable of run ids to get their guids.

    Returns:
        A list of guids for the supplied run_ids.

    """

    guids: list[str] = []

    for run_id in run_ids:
        run_id_guid = get_guid_from_run_id(conn=conn, run_id=run_id)
        if run_id_guid is not None:
            guids.append(run_id_guid)
        else:
            raise RuntimeError(f"run id {run_id} does not exist in the database")

    return guids


def finish_experiment(conn: AtomicConnection, exp_id: int) -> None:
    """Finish experiment

    Args:
        conn: database connection
        exp_id: the id of the experiment

    """
    query = """
    UPDATE experiments SET end_time=? WHERE exp_id=?;
    """
    atomic_transaction(conn, query, time.time(), exp_id)


def get_run_counter(conn: AtomicConnection, exp_id: int) -> int:
    """Get the experiment run counter

    Args:
        conn: the connection to the sqlite database
        exp_id: experiment identifier

    Returns:
        the experiment run counter
    Raises:
        RuntimeError if the experiment is not found.

    """
    counter = select_one_where(
        conn, "experiments", "run_counter", where_column="exp_id", where_value=exp_id
    )
    assert isinstance(counter, int)
    return counter


def get_experiments(conn: AtomicConnection) -> list[int]:
    """
    Get a list of experiments

    Args:
        conn: database connection

    Returns:
        list of rows

    """
    sql = "SELECT exp_id FROM experiments"
    c = atomic_transaction(conn, sql)

    return [exp_id for (exp_id,) in c.fetchall()]


def get_matching_exp_ids(conn: AtomicConnection, **match_conditions: Any) -> list[int]:
    """
    Get exp_ids for experiments matching the match_conditions

    Raises:
        ValueError if a match_condition that is not "name", "sample_name",
        "format_string", "run_counter", "start_time", or "end_time"

    """
    valid_conditions = [
        "name",
        "sample_name",
        "start_time",
        "end_time",
        "run_counter",
        "format_string",
    ]

    for mcond in match_conditions:
        if mcond not in valid_conditions:
            raise ValueError(f"{mcond} is not a valid match condition.")

    end_time = match_conditions.get("end_time", None)
    time_eq = "=" if end_time is not None else "IS"

    sample_name = match_conditions.get("sample_name", None)
    sample_name_eq = "=" if sample_name is not None else "IS"

    query = "SELECT exp_id FROM experiments "
    for n, mcond in enumerate(match_conditions):
        if n == 0:
            query += f"WHERE {mcond} = ? "
        else:
            query += f"AND {mcond} = ? "

    # now some syntax clean-up
    if "format_string" in match_conditions:
        format_string = match_conditions["format_string"]
        query = query.replace("format_string = ?", f'format_string = "{format_string}"')
        match_conditions.pop("format_string")
    query = query.replace("end_time = ?", f"end_time {time_eq} ?")
    query = query.replace("sample_name = ?", f"sample_name {sample_name_eq} ?")

    cursor = conn.execute(query, tuple(match_conditions.values()))

    return [exp_id for (exp_id,) in cursor.fetchall()]


def get_exp_ids_from_run_ids(
    conn: AtomicConnection, run_ids: Sequence[int]
) -> list[int]:
    """
    Get the corresponding exp_id for a sequence of run_ids

    Args:
        conn: connection to the database
        run_ids: a sequence of the run_ids to get the exp_id of

    Returns:
        A list of exp_ids matching the run_ids

    """
    sql_placeholders = sql_placeholder_string(len(run_ids))
    exp_id_query = f"""
                    SELECT exp_id
                    FROM runs
                    WHERE run_id IN {sql_placeholders}
                    """
    cursor = conn.cursor()
    cursor.execute(exp_id_query, run_ids)
    rows = cursor.fetchall()

    return [exp_id for row in rows for exp_id in row]


def get_last_experiment(conn: AtomicConnection) -> int | None:
    """
    Return last started experiment id

    Returns None if there are no experiments in the database
    """
    query = "SELECT MAX(exp_id) FROM experiments"
    c = atomic_transaction(conn, query)
    return c.fetchall()[0][0]


def get_runs(conn: AtomicConnection, exp_id: int | None = None) -> list[int]:
    """Get a list of runs.

    Args:
        conn: database connection
        exp_id: id of the experiment to look inside.
            If None all experiments will be included

    Returns:
        list of rows

    """
    with atomic(conn) as atomic_conn:
        if exp_id:
            sql = """
            SELECT run_id FROM runs
            WHERE exp_id = ?
            """
            c = transaction(atomic_conn, sql, exp_id)
        else:
            sql = "SELECT run_id FROM runs"
            c = transaction(atomic_conn, sql)

    return [run_id for (run_id,) in c.fetchall()]


def get_last_run(conn: AtomicConnection, exp_id: int | None = None) -> int | None:
    """
    Get run_id of the last run in experiment with exp_id

    Args:
        conn: connection to use for the query
        exp_id: id of the experiment to look inside.
            If None all experiments will be included

    Returns:
        the integer id of the last run or None if there are not runs in the
        experiment

    """
    if exp_id is not None:
        query = """
            SELECT run_id, max(run_timestamp), exp_id
            FROM runs
            WHERE exp_id = ?;
            """
        c = atomic_transaction(conn, query, exp_id)
    else:
        query = """
            SELECT run_id, max(run_timestamp)
            FROM runs
            """
        c = atomic_transaction(conn, query)
    return one(c, "run_id")


def run_exists(conn: AtomicConnection, run_id: int) -> bool:
    # the following query always returns a single tuple with an integer
    # value of `1` or `0` for existing and non-existing run_id in the database
    query = """
    SELECT EXISTS(
        SELECT 1
        FROM runs
        WHERE run_id = ?
        LIMIT 1
    );
    """
    res: tuple[int] = atomic_transaction(conn, query, run_id).fetchone()
    return bool(res[0])


def format_table_name(fmt_str: str, name: str, exp_id: int, run_counter: int) -> str:
    """
    Format the format_string into a table name

    Args:
        fmt_str: a valid format string
        name: the run name
        exp_id: the experiment ID
        run_counter: the intra-experiment runnumber of this run

    """
    table_name = fmt_str.format(name, exp_id, run_counter)
    _validate_table_name(table_name)  # raises if table_name not valid
    return table_name


def _insert_run(
    conn: AtomicConnection,
    exp_id: int,
    name: str,
    guid: str,
    captured_run_id: int | None = None,
    captured_counter: int | None = None,
    parent_dataset_links: str = "[]",
    description: RunDescriber | None = None,
) -> tuple[int, str, int]:
    # get run counter and formatter from experiments
    run_counter, format_string = select_many_where(
        conn,
        "experiments",
        "run_counter",
        "format_string",
        where_column="exp_id",
        where_value=exp_id,
    )
    assert isinstance(run_counter, int)
    assert isinstance(format_string, str)
    newly_created_run = False

    if (captured_run_id is not None and captured_counter is None) or (
        captured_counter is not None and captured_run_id is None
    ):
        raise ValueError(
            "Either both of captured_counter and captured_run_id should be set or none of them."
        )

    run_counter += 1
    if captured_counter is None:
        newly_created_run = True
        with atomic(conn) as atomic_conn:
            query = """
            SELECT
                max(result_counter)
            FROM
                runs
            WHERE
                exp_id = ?"""
            curr = transaction(atomic_conn, query, exp_id)
            existing_captured_counter = one(curr, 0)
            if existing_captured_counter is not None:
                captured_counter = existing_captured_counter + 1
            else:
                captured_counter = run_counter

    if captured_run_id is None:
        with atomic(conn) as atomic_conn:
            query = """
            SELECT
                max(run_id)
            FROM
                runs"""
            curr = transaction(atomic_conn, query)
            existing_captured_run_id = one(curr, 0)
        if existing_captured_run_id is not None:
            captured_run_id = existing_captured_run_id + 1
        else:
            captured_run_id = 1

    formatted_name = format_table_name(format_string, "results", exp_id, run_counter)
    table = "runs"

    if description is None:
        description = RunDescriber(InterDependencies_())
    desc_str = serial.to_json_for_storage(description)
    legacy_param_specs = new_to_old(description.interdeps).paramspecs
    # we need to use legacy_param_specs here not because we need the features
    # of legacy param specs but because they should be inserted into the layouts
    # table and parameter column in the same order as data_set.add_parameter does
    # specifically dependencies should come before the dependent parameters for links
    # in the layout table to work correctly

    with atomic(conn) as atomic_conn:
        if legacy_param_specs:
            query = f"""
            INSERT INTO {table}
                (name,
                 exp_id,
                 guid,
                 result_table_name,
                 result_counter,
                 run_timestamp,
                 parameters,
                 is_completed,
                 run_description,
                 captured_run_id,
                 captured_counter,
                 parent_datasets)
            VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?)
            """
            curr = transaction(
                atomic_conn,
                query,
                name,
                exp_id,
                guid,
                formatted_name,
                run_counter,
                None,
                ",".join(p.name for p in legacy_param_specs),
                False,
                desc_str,
                captured_run_id,
                captured_counter,
                parent_dataset_links,
            )
            run_id = curr.lastrowid
            if run_id is None:
                raise RuntimeError(f"Creation of run with guid: {guid} failed")

            _add_parameters_to_layout_and_deps(atomic_conn, run_id, *legacy_param_specs)

        else:
            query = f"""
            INSERT INTO {table}
                (name,
                 exp_id,
                 guid,
                 result_table_name,
                 result_counter,
                 run_timestamp,
                 is_completed,
                 run_description,
                 captured_run_id,
                 captured_counter,
                 parent_datasets)
            VALUES
                (?,?,?,?,?,?,?,?,?,?,?)
            """
            curr = transaction(
                conn,
                query,
                name,
                exp_id,
                guid,
                formatted_name,
                run_counter,
                None,
                False,
                desc_str,
                captured_run_id,
                captured_counter,
                parent_dataset_links,
            )

    run_id = curr.lastrowid
    if run_id is None:
        raise RuntimeError(f"Creation of run with guid: {guid} failed")
    if newly_created_run:
        if captured_run_id != run_id:
            warnings.warn(
                "captured_run_id was not equal to run_id, this is unexpected and will be an error in the future."
            )
        if captured_counter != run_counter:
            warnings.warn(
                "captured_counter was not equal to result_counter, this is unexpected and will be an error in the future."
            )
    return run_counter, formatted_name, run_id


def _update_experiment_run_counter(
    conn: AtomicConnection, exp_id: int, run_counter: int
) -> None:
    query = """
    UPDATE experiments
    SET run_counter = ?
    WHERE exp_id = ?
    """
    atomic_transaction(conn, query, run_counter, exp_id)


def _get_parameters(conn: AtomicConnection, run_id: int) -> list[ParamSpec]:
    """
    Get the list of param specs for run

    Args:
        conn: the connection to the sqlite database
        run_id: The id of the run

    Returns:
        A list of param specs for this run

    """

    sql = """
    SELECT parameter FROM layouts
    WHERE run_id = ?
    """
    c = conn.execute(sql, (run_id,))
    return [
        _get_paramspec(conn, run_id, param_name)
        for (param_name,) in many_many(c, "parameter")
    ]


def _get_paramspec(conn: AtomicConnection, run_id: int, param_name: str) -> ParamSpec:
    """
    Get the ParamSpec object for the given parameter name
    in the given run

    Args:
        conn: Connection to the database
        run_id: The run id
        param_name: The name of the parameter

    """

    # get table name
    sql = """
    SELECT result_table_name FROM runs
    WHERE run_id = ?
    """
    c = conn.execute(sql, (run_id,))
    result_table_name = one(c, "result_table_name")

    # get the data type
    sql = f"""
    PRAGMA TABLE_INFO("{result_table_name}")
    """
    c = c.execute(sql)
    description = get_description_map(c)
    param_type = None
    for row in c.fetchall():
        if row[description["name"]] == param_name:
            param_type = row[description["type"]]
            break

    # get everything else

    sql = """
    SELECT layout_id, run_id, parameter, label, unit, inferred_from FROM layouts
    WHERE parameter = ? and run_id = ?
    """
    c = c.execute(sql, (param_name, run_id))
    (layout_id, _, _, label, unit, inferred_from_string) = many(
        c, "layout_id", "run_id", "parameter", "label", "unit", "inferred_from"
    )

    if inferred_from_string:
        inferred_from = inferred_from_string.split(", ")
    else:
        inferred_from = []

    deps = _get_dependencies(conn, layout_id)
    depends_on: list[str] | None
    if len(deps) == 0:
        depends_on = None
    else:
        dps: list[int] = [dp[0] for dp in deps]
        ax_nums: list[int] = [dp[1] for dp in deps]
        depends_on = []
        for _, dp in sorted(zip(ax_nums, dps)):
            sql = """
            SELECT parameter FROM layouts
            WHERE layout_id = ?
            """
            c = conn.execute(sql, (dp,))
            depends_on.append(one(c, "parameter"))

    assert param_type is not None
    parspec = ParamSpec(param_name, param_type, label, unit, inferred_from, depends_on)
    return parspec


def update_run_description(
    conn: AtomicConnection, run_id: int, description: str
) -> None:
    """
    Update the run_description field for the given run_id. The description
    string must be a valid JSON string representation of a RunDescriber object
    """
    try:
        serial.from_json_to_current(description)
    except Exception as e:
        raise ValueError(
            "Invalid description string. Must be a JSON string "
            "representation of a RunDescriber object."
        ) from e

    _update_run_description(conn, run_id, description)


def _update_run_description(
    conn: AtomicConnection, run_id: int, description: str
) -> None:
    """
    Update the run_description field for the given run_id. The description
    string is NOT validated.
    """
    sql = """
          UPDATE runs
          SET run_description = ?
          WHERE run_id = ?
          """
    with atomic(conn) as atomic_conn:
        atomic_conn.cursor().execute(sql, (description, run_id))


def update_parent_datasets(conn: AtomicConnection, run_id: int, links_str: str) -> None:
    """
    Update (i.e. overwrite) the parent_datasets field for the given run_id
    """
    if not is_column_in_table(conn, "runs", "parent_datasets"):
        insert_column(conn, "runs", "parent_datasets")

    sql = """
          UPDATE runs
          SET parent_datasets = ?
          WHERE run_id = ?
          """
    with atomic(conn) as atomic_conn:
        atomic_conn.cursor().execute(sql, (links_str, run_id))


def set_run_timestamp(
    conn: AtomicConnection, run_id: int, timestamp: float | None = None
) -> None:
    """
    Set the run_timestamp for the run with the given run_id. If the
    run_timestamp has already been set, a RuntimeError is raised.

    Args:
        conn: database connection
        run_id: id of the run to mark complete
        timestamp: time stamp for completion. If None the function will
            automatically get the current time.

    """

    query = """
            SELECT run_timestamp
            FROM runs
            WHERE run_id = ?
            """
    cmd = """
          UPDATE runs
          SET run_timestamp = ?
          WHERE run_id = ?
          """

    with atomic(conn) as atomic_conn:
        c = atomic_conn.cursor()
        old_timestamp = one(c.execute(query, (run_id,)), "run_timestamp")
        if old_timestamp is not None:
            raise RuntimeError(
                "Can not set run_timestamp; it has already "
                f"been set to: {old_timestamp}"
            )
        else:
            if timestamp is None:
                timestamp = time.time()
            c.execute(cmd, (timestamp, run_id))
            log.info(f"Set the run_timestamp of run_id {run_id} to {timestamp}")


def add_parameter(
    *parameter: ParamSpec,
    conn: AtomicConnection,
    run_id: int,
    insert_into_results_table: bool,
) -> None:
    """
    Add parameters to the dataset

    This will update the layouts and dependencies tables

    NOTE: two parameters with the same name are not allowed

    Args:
        conn: the connection to the sqlite database
        run_id: id ot the run to add parameters to
        insert_into_results_table: Should the parameters be added as columns to the
           results table?
        parameter: the list of ParamSpecs for parameters to add

    """
    with atomic(conn) as atomic_conn:
        sql = "SELECT result_table_name FROM runs WHERE run_id=?"
        formatted_name = one(transaction(atomic_conn, sql, run_id), "result_table_name")

        p_names = []
        for p in parameter:
            if insert_into_results_table:
                insert_column(atomic_conn, formatted_name, p.name, p.type)
            p_names.append(p.name)
        # get old parameters column from run table
        sql = """
        SELECT parameters FROM runs
        WHERE run_id=?
        """
        with atomic(atomic_conn) as atomic_conn_1:
            c = transaction(atomic_conn_1, sql, run_id)
        old_parameters = one(c, "parameters")
        if old_parameters:
            new_parameters = ",".join([old_parameters, *p_names])
        else:
            new_parameters = ",".join(p_names)
        sql = "UPDATE runs SET parameters=? WHERE run_id=?"
        with atomic(atomic_conn) as atomic_conn_1:
            transaction(atomic_conn_1, sql, new_parameters, run_id)

        # Update the layouts table
        c = _add_parameters_to_layout_and_deps(atomic_conn, run_id, *parameter)


def _add_parameters_to_layout_and_deps(
    conn: AtomicConnection, run_id: int, *parameter: ParamSpec
) -> sqlite3.Cursor:
    layout_args: list[int | str] = []
    for p in parameter:
        layout_args.append(run_id)
        layout_args.append(p.name)
        layout_args.append(p.label)
        layout_args.append(p.unit)
        layout_args.append(p.inferred_from)
    rowplaceholder = "(?, ?, ?, ?, ?)"
    placeholder = ",".join([rowplaceholder] * len(parameter))
    sql = f"""
    INSERT INTO layouts (run_id, parameter, label, unit, inferred_from)
    VALUES {placeholder}
    """

    with atomic(conn) as atomic_conn:
        c = transaction(atomic_conn, sql, *layout_args)

        for p in parameter:
            if p.depends_on != "":
                layout_id = _get_layout_id(atomic_conn, p, run_id)

                deps = p.depends_on.split(", ")
                for ax_num, dp in enumerate(deps):
                    sql = """
                    SELECT layout_id FROM layouts
                    WHERE run_id=? and parameter=?;
                    """

                    c = transaction(atomic_conn, sql, run_id, dp)
                    dep_ind = one(c, "layout_id")

                    sql = """
                    INSERT INTO dependencies (dependent, independent, axis_num)
                    VALUES (?,?,?)
                    """

                    c = transaction(atomic_conn, sql, layout_id, dep_ind, ax_num)
    return c


def _validate_table_name(table_name: str) -> bool:
    valid = True
    for i in table_name:
        if unicodedata.category(i) not in _unicode_categories:
            valid = False
            raise RuntimeError(f"Invalid table name {table_name} starting at {i}")
    return valid


def _create_run_table(
    conn: AtomicConnection,
    formatted_name: str,
    parameters: Sequence[ParamSpecBase] | None = None,
    values: VALUES | None = None,
) -> None:
    """Create run table with formatted_name as name

    Args:
        conn: database connection
        formatted_name: the name of the table to create
        parameters: Parameters to insert in the table.
        values: Values for the parameters above.

    """
    _validate_table_name(formatted_name)

    with atomic(conn) as atomic_conn:
        if parameters and values:
            _parameters = ",".join(p.sql_repr() for p in parameters)
            query = f"""
            CREATE TABLE "{formatted_name}" (
                id INTEGER PRIMARY KEY,
                {_parameters}
            );
            """
            transaction(atomic_conn, query)
            # now insert values
            insert_values(
                atomic_conn, formatted_name, [p.name for p in parameters], values
            )
        elif parameters:
            _parameters = ",".join(p.sql_repr() for p in parameters)
            query = f"""
            CREATE TABLE "{formatted_name}" (
                id INTEGER PRIMARY KEY,
                {_parameters}
            );
            """
            transaction(atomic_conn, query)
        else:
            query = f"""
            CREATE TABLE "{formatted_name}" (
                id INTEGER PRIMARY KEY
            );
            """
            transaction(atomic_conn, query)


def create_run(
    conn: AtomicConnection,
    exp_id: int,
    name: str,
    guid: str,
    parameters: Sequence[ParamSpec] | None = None,
    values: Sequence[Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    captured_run_id: int | None = None,
    captured_counter: int | None = None,
    parent_dataset_links: str = "[]",
    create_run_table: bool = True,
    snapshot_raw: str | None = None,
    description: RunDescriber | None = None,
) -> tuple[int, int, str | None]:
    """Create a single run for the experiment.


    This will register the run in the runs table, the counter in the
    experiments table and optionally create a new table with the formatted name.

    Note that it is an error to supply both Parameters and RunDescriber

    Args:
        conn: the connection to the sqlite database
        exp_id: the experiment id we want to create the run into
        name: a friendly name for this run
        guid: the guid adhering to our internal guid format
        parameters: optional list of parameters this run has (as ParamSpec objects).
           This is not recommended, please use description instead.
        values:  optional list of values for the parameters
        metadata: optional metadata dictionary
        captured_run_id: The run_id this data was originally captured with.
            Should only be supplied when inserting an already completed run
            from another database into this database. Otherwise leave as None.
        captured_counter: The counter this data was originally captured with.
            Should only be supplied when inserting an already completed run
            from another database into this database. Otherwise leave as None.
        parent_dataset_links: A JSON string of the parent-child dataset links for the run.
        create_run_table: Should we create a table to insert the run into.
        snapshot_raw: Raw string of the snapshot to add to the run.
        description: An optional RunDescriber

    Returns:
        run_counter: the id of the newly created run (not unique)
        run_id: the row id of the newly created run
        formatted_name: the name of the newly created table

    """
    formatted_name: str | None

    if parameters is not None and description is not None:
        raise RuntimeError(
            "Passing both parameters and description to create_run is not supported."
        )
    if parameters is not None:
        warnings.warn(
            "Passing parameters to create_run is deprecated and will "
            "be removed in the future",
            stacklevel=2,
        )
        description = RunDescriber(old_to_new(v0.InterDependencies(*parameters)))
    elif description is None:
        description = RunDescriber(InterDependencies_())

    with atomic(conn):
        run_counter, formatted_name, run_id = _insert_run(
            conn=conn,
            exp_id=exp_id,
            name=name,
            guid=guid,
            captured_run_id=captured_run_id,
            captured_counter=captured_counter,
            parent_dataset_links=parent_dataset_links,
            description=description,
        )
        if metadata:
            add_data_to_dynamic_columns(conn, run_id, metadata)
        if snapshot_raw:
            add_data_to_dynamic_columns(conn, run_id, {"snapshot": snapshot_raw})
        _update_experiment_run_counter(conn, exp_id, run_counter)
        if create_run_table:
            _create_run_table(
                conn, formatted_name, description.interdeps.paramspecs, values
            )
        else:
            formatted_name = None
    log.info(
        "Created run with guid: %s in database %s",
        guid,
        conn.path_to_dbfile,
        extra={"qcodes_guid": guid, "path_to_db": conn.path_to_dbfile},
    )
    return run_counter, run_id, formatted_name


def get_run_description(conn: AtomicConnection, run_id: int) -> str:
    """
    Return the (JSON string) run description of the specified run
    """
    rds = select_one_where(conn, "runs", "run_description", "run_id", run_id)
    assert isinstance(rds, str)
    return rds


def get_parent_dataset_links(conn: AtomicConnection, run_id: int) -> str:
    """
    Return the (JSON string) of the parent-child dataset links for the
    specified run
    """

    # We cannot in general trust that NULLs will not appear in the column,
    # even if the column is present in the runs table.

    link_str: str
    maybe_link_str: str | None

    if not is_column_in_table(conn, "runs", "parent_datasets"):
        maybe_link_str = None
    else:
        maybe_mayby_link_str = select_one_where(
            conn, "runs", "parent_datasets", "run_id", run_id
        )
        assert isinstance(maybe_mayby_link_str, (str, type(None)))
        maybe_link_str = maybe_mayby_link_str

    if maybe_link_str is None:
        link_str = "[]"
    else:
        link_str = str(maybe_link_str)

    return link_str


def get_data_by_tag_and_table_name(
    conn: AtomicConnection, tag: str, table_name: str
) -> VALUE | None:
    """
    Get data from the "tag" column for the row in "runs" table where
    "result_table_name" matches "table_name".
    Returns None if the "tag" column is missing in "runs" table.
    """
    try:
        data = select_one_where(conn, "runs", tag, "result_table_name", table_name)
    except RuntimeError as e:
        # all errors trigger an runtime error here since select_one_where is wrapped
        # in an atomic that will do a rollback
        # this probably just means that the column is not there
        # and therefore it contains no data
        if str(e.__cause__).startswith("no such column") or str(e).startswith(
            "no such column"
        ):
            data = None
        else:
            raise e
    return data


def get_metadata_from_run_id(conn: AtomicConnection, run_id: int) -> dict[str, Any]:
    """
    Get all metadata associated with the specified run
    """
    non_metadata = RUNS_TABLE_COLUMNS

    metadata = {}
    possible_tags = []

    # first fetch all columns of the runs table
    query = "PRAGMA table_info(runs)"
    cursor = conn.execute(query)
    description = get_description_map(cursor)
    for row in cursor.fetchall():
        if row[description["name"]] not in non_metadata:
            possible_tags.append(row[description["name"]])

    # and then fetch whatever metadata the run might have
    for tag in possible_tags:
        query = f"""
        SELECT "{tag}" FROM runs
        WHERE run_id = ? AND "{tag}" IS NOT NULL
        """
        cursor.execute(query, (run_id,))
        row = cursor.fetchall()
        if row != []:
            metadata[tag] = row[0][0]

    return metadata


def validate_dynamic_column_data(data: Mapping[str, Any]) -> None:
    """
    Validate the given dicts tags and values. Note that None is not a valid
    value, and keys should be valid SQLite column names
    (i.e. contain only alphanumeric characters and underscores).

    Args:
        data: the metadata mapping (tags to values)

    """
    for tag, val in data.items():
        if not tag.isidentifier():
            raise KeyError(
                f"Tag {tag} is not a valid tag. "
                "Use only alphanumeric characters and underscores!"
            )
        if val is None:
            raise ValueError(
                f"Tag {tag} has value None. That is not a valid metadata value!"
            )


def insert_data_in_dynamic_columns(
    conn: AtomicConnection, row_id: int, table_name: str, data: Mapping[str, Any]
) -> None:
    """
    Insert new data column and add values. Note that None is not a valid
    value, and keys should be valid SQLite column names
    (i.e. contain only alphanumeric characters and underscores).

    Args:
        conn: the connection to the sqlite database
        row_id: the row to add the metadata at
        table_name: the table to add to, defaults to runs
        data: A mapping from columns to data to add

    """
    validate_dynamic_column_data(data)
    for key in data.keys():
        insert_column(conn, table_name, key)
    update_columns(conn, row_id, table_name, data)


def update_columns(
    conn: AtomicConnection, row_id: int, table_name: str, data: Mapping[str, Any]
) -> None:
    """
    Updates data in columns matching the given keys (they must exist already)

    Args:
        conn: the connection to the sqlite database
        row_id: the row to add the metadata at
        table_name: the table to add to, defaults to runs
        data: the data to add

    """
    validate_dynamic_column_data(data)
    update_where(conn, table_name, "rowid", row_id, **data)


def add_data_to_dynamic_columns(
    conn: AtomicConnection,
    row_id: int,
    data: Mapping[str, Any],
    table_name: str = "runs",
) -> None:
    """
    Add columns from keys and insert values.
    (updates if exists, creates otherwise)

    Note that None is not a valid value, and keys
    should be valid SQLite column names (i.e. contain only
    alphanumeric characters and underscores).

    Args:
        conn: the connection to the sqlite database
        row_id: the row to add the metadata at
        data: the data to add
        table_name: the table to add to, defaults to runs

    """
    try:
        insert_data_in_dynamic_columns(conn, row_id, table_name, data)
    except sqlite3.OperationalError as e:
        # this means that the column already exists
        # so just insert the new value
        if str(e).startswith("duplicate"):
            update_columns(conn, row_id, table_name, data)
        else:
            raise e


def get_experiment_name_from_experiment_id(conn: AtomicConnection, exp_id: int) -> str:
    exp_name = select_one_where(conn, "experiments", "name", "exp_id", exp_id)
    assert isinstance(exp_name, str)
    return exp_name


def get_sample_name_from_experiment_id(conn: AtomicConnection, exp_id: int) -> str:
    sample_name = select_one_where(conn, "experiments", "sample_name", "exp_id", exp_id)
    assert isinstance(sample_name, (str, type(None)))
    # there may be a few cases for very old db where None is returned as a sample name
    # however, these probably do not exist in relaity outside that test so here we
    # cast to str. See test_experiments_with_NULL_sample_name
    return cast("str", sample_name)


def get_run_timestamp_from_run_id(conn: AtomicConnection, run_id: int) -> float | None:
    time_stamp = select_one_where(conn, "runs", "run_timestamp", "run_id", run_id)
    # sometimes it happens that the timestamp is saved as an integer in the database
    if isinstance(time_stamp, int):
        time_stamp = float(time_stamp)
    assert isinstance(time_stamp, (float, type(None)))
    return time_stamp


def update_GUIDs(conn: AtomicConnection) -> None:
    """
    Update all GUIDs in this database where either the location code or the
    work_station code is zero to use the location and work_station code from
    the qcodesrc.json file in home. Runs where it is not true that both codes
    are zero are skipped.
    """

    log.info("Commencing update of all GUIDs in database")

    cfg = qc.config

    location = cfg["GUID_components"]["location"]
    work_station = cfg["GUID_components"]["work_station"]

    if location == 0:
        log.warning(
            "The location is still set to the default (0). Can not "
            "proceed. Please configure the location before updating "
            "the GUIDs."
        )
        return
    if work_station == 0:
        log.warning(
            "The work_station is still set to the default (0). Can not"
            " proceed. Please configure the location before updating "
            "the GUIDs."
        )
        return

    query = "select MAX(run_id) from runs"
    c = atomic_transaction(conn, query)
    no_of_runs = c.fetchall()[0][0]

    # now, there are four actions we can take

    def _both_nonzero(run_id: int, *args: Any) -> None:
        log.info(f"Run number {run_id} already has a valid GUID, skipping.")

    def _location_only_zero(run_id: int, *args: Any) -> None:
        log.warning(
            f"Run number {run_id} has a zero (default) location "
            "code, but a non-zero work station code. Please manually "
            "resolve this, skipping the run now."
        )

    def _workstation_only_zero(run_id: int, *args: Any) -> None:
        log.warning(
            f"Run number {run_id} has a zero (default) work station"
            " code, but a non-zero location code. Please manually "
            "resolve this, skipping the run now."
        )

    def _both_zero(
        run_id: int, conn: AtomicConnection, guid_comps: dict[str, Any]
    ) -> None:
        guid_str = build_guid_from_components(guid_comps)
        with atomic(conn) as atomic_conn:
            sql = """
                   UPDATE runs
                   SET guid = ?
                   WHERE run_id = ?
                   """
            atomic_conn.execute(sql, (guid_str, run_id))

        log.info(f"Succesfully updated run number {run_id}.")

    actions: dict[
        tuple[bool, bool], Callable[[int, AtomicConnection, dict[str, Any]], None]
    ]
    actions = {
        (True, True): _both_zero,
        (False, True): _workstation_only_zero,
        (True, False): _location_only_zero,
        (False, False): _both_nonzero,
    }

    for run_id in range(1, no_of_runs + 1):
        guid_str = get_guid_from_run_id(conn, run_id)
        assert guid_str is not None
        guid_comps = parse_guid(guid_str)
        old_loc = guid_comps["location"]
        old_ws = guid_comps["work_station"]

        guid_comps["location"] = location
        guid_comps["work_station"] = work_station

        log.info(f"Updating run number {run_id}...")
        actions[(old_loc == 0, old_ws == 0)](run_id, conn, guid_comps)


def remove_trigger(conn: AtomicConnection, trigger_id: str) -> None:
    """
    Removes a trigger with a given id if it exists.

    Note that this transaction is not atomic!

    Args:
        conn: database connection object
        trigger_id: id of the trigger

    """
    transaction(conn, f"DROP TRIGGER IF EXISTS {trigger_id};")


def load_new_data_for_rundescriber(
    conn: AtomicConnection,
    table_name: str,
    rundescriber: RunDescriber,
    read_status: Mapping[str, int],
) -> tuple[dict[str, dict[str, npt.NDArray]], dict[str, int]]:
    """
    Load all new data for a given rundesciber since the rows given by read_status.

    Args:
        conn: The connection to the sqlite database
        table_name: The name of the table the data is stored in
        rundescriber: The rundescriber that describes the run
        read_status: Mapping from dependent parameter name to number of rows
          read from the db previously.

    Returns:
        new data and an updated number of rows read.

    """

    parameters = tuple(ps.name for ps in rundescriber.interdeps.top_level_parameters)
    updated_read_status: dict[str, int] = dict(read_status)
    new_data_dict: dict[str, dict[str, npt.NDArray]] = {}

    for meas_parameter in parameters:
        start = read_status.get(meas_parameter, 0) + 1
        new_data, n_rows_read = get_parameter_data_for_one_paramtree(
            conn,
            table_name,
            rundescriber=rundescriber,
            output_param=meas_parameter,
            start=start,
            end=None,
            callback=None,
        )
        new_data_dict[meas_parameter] = new_data
        updated_read_status[meas_parameter] = start + n_rows_read - 1
    return new_data_dict, updated_read_status


class ExperimentAttributeDict(TypedDict):
    exp_id: int
    name: str
    sample_name: str
    start_time: float
    end_time: float | None
    format_string: str


def get_experiment_attributes_by_exp_id(
    conn: AtomicConnection, exp_id: int
) -> ExperimentAttributeDict:
    """
    Return a dict of all attributes describing an experiment from the exp_id.

    Args:
        conn: The connection to the sqlite database
        exp_id: the id of the experiment

    Returns:
        A dictionary of the experiment attributes.

    """
    exp_attr_names = ["name", "sample_name", "start_time", "end_time", "format_string"]

    exp_attr_vals = select_many_where(
        conn, "experiments", *exp_attr_names, where_column="exp_id", where_value=exp_id
    )

    temp_exp_attrs = dict(zip(exp_attr_names, exp_attr_vals))
    start_time = temp_exp_attrs["start_time"]
    assert isinstance(start_time, float)
    end_time = temp_exp_attrs["end_time"]
    assert isinstance(end_time, (float, type(None)))

    exp_attrs: ExperimentAttributeDict = {
        "name": str(temp_exp_attrs["name"]),
        "sample_name": str(temp_exp_attrs["sample_name"]),
        "start_time": start_time,
        "end_time": end_time,
        "format_string": str(temp_exp_attrs["format_string"]),
        "exp_id": exp_id,
    }

    return exp_attrs


def _populate_results_table(
    source_conn: AtomicConnection,
    target_conn: AtomicConnection,
    source_table_name: str,
    target_table_name: str,
) -> None:
    """
    Copy over all the entries of the results table
    """
    get_data_query = f"""
                     SELECT *
                     FROM "{source_table_name}"
                     """

    source_cursor = source_conn.cursor()
    target_cursor = target_conn.cursor()

    for row in source_cursor.execute(get_data_query):
        column_names = ",".join(
            d[0] for d in source_cursor.description[1:]
        )  # the first key is "id"
        values = tuple(val for val in row[1:])
        value_placeholders = sql_placeholder_string(len(values))
        insert_data_query = f"""
        INSERT INTO "{target_table_name}"
        ({column_names})
        values {value_placeholders}
        """
        target_cursor.execute(insert_data_query, values)


def _rewrite_timestamps(
    target_conn: AtomicConnection,
    target_run_id: int,
    correct_run_timestamp: float | None,
    correct_completed_timestamp: float | None,
) -> None:
    """
    Update the timestamp to match the original one
    """
    query = """
            UPDATE runs
            SET run_timestamp = ?
            WHERE run_id = ?
            """
    cursor = target_conn.cursor()
    cursor.execute(query, (correct_run_timestamp, target_run_id))

    query = """
            UPDATE runs
            SET completed_timestamp = ?
            WHERE run_id = ?
            """
    cursor = target_conn.cursor()
    cursor.execute(query, (correct_completed_timestamp, target_run_id))


class RawRunAttributesDict(TypedDict):
    run_id: int
    counter: int
    captured_run_id: int
    captured_counter: int
    experiment: ExperimentAttributeDict
    name: str
    run_timestamp: float | None
    completed_timestamp: float | None
    metadata: dict[str, Any]
    parent_dataset_links: str
    run_description: str
    snapshot: str | None


def get_raw_run_attributes(
    conn: AtomicConnection, guid: str
) -> RawRunAttributesDict | None:
    run_id = get_runid_from_guid(conn, guid)

    if run_id is None:
        return None

    exp_id = get_exp_ids_from_run_ids(conn, [run_id])[0]
    experiment = get_experiment_attributes_by_exp_id(conn, exp_id)

    counter = select_one_where(conn, "runs", "result_counter", "guid", guid)
    assert isinstance(counter, int)
    captured_run_id = select_one_where(conn, "runs", "captured_run_id", "guid", guid)
    assert isinstance(captured_run_id, int)

    captured_counter = select_one_where(conn, "runs", "captured_counter", "guid", guid)
    assert isinstance(captured_counter, int)

    name = select_one_where(conn, "runs", "name", "guid", guid)
    assert isinstance(name, str)

    rawsnapshot = select_one_where(conn, "runs", "snapshot", "guid", guid)
    assert isinstance(rawsnapshot, (str, type(None)))
    output: RawRunAttributesDict = {
        "run_id": run_id,
        "experiment": experiment,
        "counter": counter,
        "captured_run_id": captured_run_id,
        "captured_counter": captured_counter,
        "name": name,
        "run_timestamp": get_run_timestamp_from_run_id(conn, run_id),
        "completed_timestamp": get_completed_timestamp_from_run_id(conn, run_id),
        "metadata": get_metadata_from_run_id(conn, run_id),
        "parent_dataset_links": get_parent_dataset_links(conn, run_id),
        "run_description": get_run_description(conn, run_id),
        "snapshot": rawsnapshot,
    }

    return output


def raw_time_to_str_time(
    raw_timestamp: float | None, fmt: str = "%Y-%m-%d %H:%M:%S"
) -> str | None:
    if raw_timestamp is None:
        return None
    else:
        return time.strftime(fmt, time.localtime(raw_timestamp))


def _check_if_table_found(conn: AtomicConnection, table_name: str) -> bool:
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    cursor = conn.cursor()
    return not many_many(cursor.execute(query, (table_name,)), "name") == []


def _get_result_table_name_by_guid(conn: AtomicConnection, guid: str) -> str:
    sql = "SELECT result_table_name FROM runs WHERE guid=?"
    formatted_name = one(transaction(conn, sql, guid), "result_table_name")
    return formatted_name
