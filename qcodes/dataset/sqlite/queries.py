"""
This module contains useful SQL queries and their combinations which are
specific to the domain of QCoDeS database.
"""
import logging
import sqlite3
import time
import unicodedata
import warnings
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Union, cast)
from copy import copy
import numpy as np
from numpy import VisibleDeprecationWarning

import qcodes as qc
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning import serialization as serial
from qcodes.dataset.descriptions.versioning import v0
from qcodes.dataset.descriptions.versioning.converters import old_to_new
from qcodes.dataset.guids import generate_guid, parse_guid
from qcodes.dataset.sqlite.connection import (ConnectionPlus, atomic,
                                              atomic_transaction, transaction)
from qcodes.dataset.sqlite.query_helpers import (VALUES, insert_column,
                                                 insert_values,
                                                 is_column_in_table, many,
                                                 many_many, one,
                                                 select_many_where,
                                                 select_one_where,
                                                 sql_placeholder_string,
                                                 update_where)
from qcodes.utils.deprecate import deprecate

log = logging.getLogger(__name__)


_unicode_categories = ('Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nd', 'Pc', 'Pd', 'Zs')


# in the current version, these are the standard columns of the "runs" table
# Everything else is metadata
RUNS_TABLE_COLUMNS = ["run_id", "exp_id", "name", "result_table_name",
                      "result_counter", "run_timestamp", "completed_timestamp",
                      "is_completed", "parameters", "guid",
                      "run_description", "snapshot", "parent_datasets",
                      "captured_run_id", "captured_counter"]


def is_run_id_in_database(conn: ConnectionPlus,
                          *run_ids: int) -> Dict[int, bool]:
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
    run_ids = np.unique(run_ids)
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


def _build_data_query(table_name: str,
                      columns: List[str],
                      start: Optional[int] = None,
                      end: Optional[int] = None,
                      ) -> str:

    _columns = ",".join(columns)
    query = f"""
            SELECT {_columns}
            FROM "{table_name}"
            """

    start_specified = start is not None
    end_specified = end is not None

    where = ' WHERE' if start_specified or end_specified else ''
    start_condition = f' rowid >= {start}' if start_specified else ''
    end_condition = f' rowid <= {end}' if end_specified else ''
    and_ = ' AND' if start_specified and end_specified else ''

    query += where + start_condition + and_ + end_condition
    return query


@deprecate('This method does not accurately represent the dataset.',
               'Use `get_parameter_data` instead.')
def get_data(conn: ConnectionPlus,
             table_name: str,
             columns: List[str],
             start: Optional[int] = None,
             end: Optional[int] = None,
             ) -> List[List[Any]]:
    """
    Get data from the columns of a table.
    Allows to specify a range of rows (1-based indexing, both ends are
    included).

    Args:
        conn: database connection
        table_name: name of the table
        columns: list of columns
        start: start of range; if None, then starts from the top of the table
        end: end of range; if None, then ends at the bottom of the table

    Returns:
        the data requested in the format of list of rows of values
    """
    if len(columns) == 0:
        warnings.warn(
            'get_data: requested data without specifying parameters/columns.'
            'Returning empty list.'
        )
        return [[]]
    query = _build_data_query(table_name, columns, start, end)
    c = atomic_transaction(conn, query)
    res = many_many(c, *columns)

    return res


def get_parameter_data(conn: ConnectionPlus,
                       table_name: str,
                       columns: Sequence[str] = (),
                       start: Optional[int] = None,
                       end: Optional[int] = None) -> \
        Dict[str, Dict[str, np.ndarray]]:
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
    """
    rundescriber = get_rundescriber_from_result_table_name(conn, table_name)

    output = {}
    if len(columns) == 0:
        columns = [ps.name for ps in rundescriber.interdeps.non_dependencies]

    # loop over all the requested parameters
    for output_param in columns:
        output[output_param] = get_shaped_parameter_data_for_one_paramtree(
            conn,
            table_name,
            rundescriber,
            output_param,
            start,
            end)
    return output


def get_shaped_parameter_data_for_one_paramtree(
        conn: ConnectionPlus,
        table_name: str,
        rundescriber: RunDescriber,
        output_param: str,
        start: Optional[int],
        end: Optional[int]
) -> Dict[str, np.ndarray]:
    """
    Get the data for a parameter tree and reshape it according to the
    metadata about the dataset. This will only reshape the loaded data if
    the number of points in the loaded data matches the expected number of
    points registered in the metadata.
    If there are more measured datapoints
    than expected a warning will be given.
    """

    one_param_output, _ = get_parameter_data_for_one_paramtree(
        conn,
        table_name,
        rundescriber,
        output_param,
        start,
        end
    )
    if rundescriber.shapes is not None:
        shape = rundescriber.shapes.get(output_param)

        if shape is not None:
            total_len_shape = np.prod(shape)
            for name, paramdata in one_param_output.items():
                total_data_shape = np.prod(paramdata.shape)
                if total_data_shape == total_len_shape:
                    one_param_output[name] = paramdata.reshape(shape)
                elif total_data_shape > total_len_shape:
                    log.warning(f"Tried to set data shape for {name} in "
                                f"dataset {output_param} "
                                f"from metadata when "
                                f"loading but found inconsistent lengths "
                                f"{total_data_shape} and {total_len_shape}")
    return one_param_output


def get_rundescriber_from_result_table_name(
        conn: ConnectionPlus,
        result_table_name: str
) -> RunDescriber:
    sql = """
    SELECT run_id FROM runs WHERE result_table_name = ?
    """
    c = atomic_transaction(conn, sql, result_table_name)
    run_id = one(c, 'run_id')
    rd = serial.from_json_to_current(get_run_description(conn, run_id))
    return rd


def get_interdeps_from_result_table_name(conn: ConnectionPlus, result_table_name: str) -> InterDependencies_:
    rd = get_rundescriber_from_result_table_name(conn, result_table_name)
    interdeps = rd.interdeps
    return interdeps


def get_parameter_data_for_one_paramtree(
        conn: ConnectionPlus,
        table_name: str,
        rundescriber: RunDescriber,
        output_param: str,
        start: Optional[int],
        end: Optional[int]
) -> Tuple[Dict[str, np.ndarray], int]:
    interdeps = rundescriber.interdeps
    data, paramspecs, n_rows = _get_data_for_one_param_tree(
        conn, table_name, interdeps, output_param, start, end
    )
    if not paramspecs[0].name == output_param:
        raise ValueError("output_param should always be the first "
                         "parameter in a parameter tree. It is not")
    _expand_data_to_arrays(data, paramspecs)

    param_data = {}
    # Benchmarking shows that transposing the data with python types is
    # faster than transposing the data using np.array.transpose
    res_t = map(list, zip(*data))

    for paramspec, column_data in zip(paramspecs, res_t):
        try:
            if paramspec.type == "numeric":
                # there is no reliable way to
                # tell the difference between a float and and int loaded
                # from sqlite numeric columns so always fall back to float
                dtype: Optional[type] = np.float64
            else:
                dtype = None
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=VisibleDeprecationWarning,
                    message="Creating an ndarray from ragged nested sequences"
                )
                # numpy warns here and coming versions
                # will eventually raise
                # for ragged arrays if you don't explicitly set
                # dtype=object
                # It is time consuming to detect ragged arrays here
                # and it is expected to be a relatively rare situation
                # so fallback to object if the regular dtype fail
                param_data[paramspec.name] = np.array(column_data, dtype=dtype)
        except:
            # Not clear which error to catch here. This will only be clarified
            # once numpy actually starts to raise here.
            param_data[paramspec.name] = np.array(column_data, dtype=np.object)
    return param_data, n_rows


def _expand_data_to_arrays(data: List[List[Any]], paramspecs: Sequence[ParamSpecBase]) -> None:
    types = [param.type for param in paramspecs]
    # if we have array type parameters expand all other parameters
    # to arrays
    if 'array' in types and ('numeric' in types or 'text' in types
                             or 'complex' in types):
        first_array_element = types.index('array')
        numeric_elms = [i for i, x in enumerate(types)
                        if x == "numeric"]
        complex_elms = [i for i, x in enumerate(types)
                        if x == 'complex']
        text_elms = [i for i, x in enumerate(types)
                     if x == "text"]
        for row in data:
            for element in numeric_elms:
                row[element] = np.full_like(row[first_array_element],
                                            row[element],
                                            dtype=np.float)
                # todo should we handle int/float types here
                # we would in practice have to perform another
                # loop to check that all elements of a given can be cast to
                # int without loosing precision before choosing an integer
                # representation of the array
            for element in complex_elms:
                row[element] = np.full_like(row[first_array_element],
                                            row[element],
                                            dtype=np.complex)
            for element in text_elms:
                strlen = len(row[element])
                row[element] = np.full_like(row[first_array_element],
                                            row[element],
                                            dtype=f'U{strlen}')


def _get_data_for_one_param_tree(conn: ConnectionPlus, table_name: str,
                                 interdeps: InterDependencies_, output_param: str,
                                 start: Optional[int], end: Optional[int]) \
        -> Tuple[List[List[Any]], List[ParamSpecBase], int]:
    output_param_spec = interdeps._id_to_paramspec[output_param]
    # find all the dependencies of this param

    dependency_params = list(interdeps.dependencies.get(output_param_spec, ()))
    dependency_names = [param.name for param in dependency_params]
    paramspecs = [output_param_spec] + dependency_params
    res = get_parameter_tree_values(conn,
                                    table_name,
                                    output_param,
                                    *dependency_names,
                                    start=start,
                                    end=end)
    n_rows = len(res)
    return res, paramspecs, n_rows


@deprecate('This method does not accurately represent the dataset.',
           'Use `get_parameter_data` instead.')
def get_values(conn: ConnectionPlus,
               table_name: str,
               param_name: str) -> List[List[Any]]:
    """
    Get the not-null values of a parameter

    Args:
        conn: Connection to the database
        table_name: Name of the table that holds the data
        param_name: Name of the parameter to get the setpoints of

    Returns:
        The values
    """
    sql = f"""
    SELECT {param_name} FROM "{table_name}"
    WHERE {param_name} IS NOT NULL
    """
    c = atomic_transaction(conn, sql)
    res = many_many(c, param_name)

    return res


def get_parameter_tree_values(conn: ConnectionPlus,
                              result_table_name: str,
                              toplevel_param_name: str,
                              *other_param_names: str,
                              start: Optional[int] = None,
                              end: Optional[int] = None) -> List[List[Any]]:
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

    Returns:
        A list of list. The outer list index is row number, the inner list
        index is parameter value (first toplevel_param, then other_param_names)
    """

    offset = max((start - 1), 0) if start is not None else 0
    limit = max((end - offset), 0) if end is not None else -1

    if start is not None and end is not None and start > end:
        limit = 0

    # Note: if we use placeholders for the SELECT part, then we get rows
    # back that have "?" as all their keys, making further data extraction
    # impossible
    #
    # Also, placeholders seem to be ignored in the WHERE X IS NOT NULL line

    columns = [toplevel_param_name] + list(other_param_names)
    columns_for_select = ','.join(columns)

    sql_subquery = f"""
                   (SELECT {columns_for_select}
                    FROM "{result_table_name}"
                    WHERE {toplevel_param_name} IS NOT NULL)
                   """
    sql = f"""
          SELECT {columns_for_select}
          FROM {sql_subquery}
          LIMIT {limit} OFFSET {offset}
          """

    cursor = conn.cursor()
    cursor.execute(sql, ())
    res = many_many(cursor, *columns)

    return res


@deprecate(alternative="get_parameter_data")
def get_setpoints(conn: ConnectionPlus,
                  table_name: str,
                  param_name: str) -> Dict[str, List[List[Any]]]:
    """
    Get the setpoints for a given dependent parameter

    Args:
        conn: Connection to the database
        table_name: Name of the table that holds the data
        param_name: Name of the parameter to get the setpoints of

    Returns:
        A list of returned setpoint values. Each setpoint return value
        is a list of lists of Any. The first list is a list of run points,
        the second list is a list of parameter values.
    """
    # TODO: We do this in no less than 5 table lookups, surely
    # this number can be reduced

    # get run_id
    sql = """
    SELECT run_id FROM runs WHERE result_table_name = ?
    """
    c = atomic_transaction(conn, sql, table_name)
    run_id = one(c, 'run_id')

    # get the parameter layout id
    sql = """
    SELECT layout_id FROM layouts
    WHERE parameter = ?
    and run_id = ?
    """
    c = atomic_transaction(conn, sql, param_name, run_id)
    layout_id = one(c, 'layout_id')

    # get the setpoint layout ids
    sql = """
    SELECT independent FROM dependencies
    WHERE dependent = ?
    """
    c = atomic_transaction(conn, sql, layout_id)
    indeps = many_many(c, 'independent')
    indeps = [idp[0] for idp in indeps]

    # get the setpoint names
    sql = f"""
    SELECT parameter FROM layouts WHERE layout_id
    IN {str(indeps).replace('[', '(').replace(']', ')')}
    """
    c = atomic_transaction(conn, sql)
    setpoint_names_temp = many_many(c, 'parameter')
    setpoint_names = [spn[0] for spn in setpoint_names_temp]
    setpoint_names = cast(List[str], setpoint_names)

    # get the actual setpoint data
    output: Dict[str, List[List[Any]]] = {}
    for sp_name in setpoint_names:
        sql = f"""
        SELECT {sp_name}
        FROM "{table_name}"
        WHERE {param_name} IS NOT NULL
        """
        c = atomic_transaction(conn, sql)
        sps = many_many(c, sp_name)
        output[sp_name] = sps

    return output


def get_runid_from_expid_and_counter(conn: ConnectionPlus, exp_id: int,
                                     counter: int) -> int:
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
    run_id = one(c, 'run_id')
    return run_id


def get_runid_from_guid(conn: ConnectionPlus, guid: str) -> Union[int, None]:
    """
    Get the run_id of a run based on the guid

    Args:
        conn: connection to the database
        guid: the guid to look up

    Returns:
        The run_id if found, else -1.

    Raises:
        RuntimeError if more than one run with the given    GUID exists
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
        run_id = -1
    elif len(rows) > 1:
        errormssg = ('Critical consistency error: multiple runs with'
                     f' the same GUID found! {len(rows)} runs have GUID '
                     f'{guid}')
        log.critical(errormssg)
        raise RuntimeError(errormssg)
    else:
        run_id = int(rows[0]['run_id'])

    return run_id


def get_guids_from_run_spec(conn: ConnectionPlus,
                            captured_run_id: Optional[int] = None,
                            captured_counter: Optional[int] = None,
                            experiment_name: Optional[str] = None,
                            sample_name: Optional[str] = None) -> List[str]:
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
    exp_ids: Optional[List[int]]
    if experiment_name is not None or sample_name is not None:
        if sample_name is not None:
            exp_query['sample_name'] = sample_name
        if experiment_name is not None:
            exp_query['name'] = experiment_name
        exp_ids = get_matching_exp_ids(conn,
                                       **exp_query)
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

    rows = cursor.fetchall()
    results = []
    for r in rows:
        results.append(r['guid'])
    return results


def _get_layout_id(conn: ConnectionPlus,
                   parameter: Union[ParamSpec, str],
                   run_id: int) -> int:
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
        raise ValueError('Wrong parameter type, must be ParamSpec or str, '
                         f'received {type(parameter)}.')

    c = atomic_transaction(conn, sql, name, run_id)
    res = one(c, 'layout_id')

    return res


def _get_dependents(conn: ConnectionPlus,
                    run_id: int) -> List[int]:
    """
    Get dependent layout_ids for a certain run_id, i.e. the layout_ids of all
    the dependent variables
    """
    sql = """
    SELECT layout_id FROM layouts
    WHERE run_id=? and layout_id in (SELECT dependent FROM dependencies)
    """
    c = atomic_transaction(conn, sql, run_id)
    res = [d[0] for d in many_many(c, 'layout_id')]
    return res


def _get_dependencies(conn: ConnectionPlus,
                      layout_id: int) -> List[List[int]]:
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
    res = many_many(c, 'independent', 'axis_num')
    return res


# Higher level Wrappers


def new_experiment(conn: ConnectionPlus,
                   name: str,
                   sample_name: str,
                   format_string: Optional[str] = "{}-{}-{}",
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
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
    return curr.lastrowid


# TODO(WilliamHPNielsen): we should remove the redundant
# is_completed
def mark_run_complete(conn: ConnectionPlus, run_id: int) -> None:
    """ Mark run complete

    Args:
        conn: database connection
        run_id: id of the run to mark complete
    """
    query = """
    UPDATE
        runs
    SET
        completed_timestamp=?,
        is_completed=?
    WHERE run_id=?;
    """
    atomic_transaction(conn, query, time.time(), True, run_id)


def completed(conn: ConnectionPlus, run_id: int) -> bool:
    """ Check if the run is complete

    Args:
        conn: database connection
        run_id: id of the run to check
    """
    return bool(select_one_where(conn, "runs", "is_completed",
                                 "run_id", run_id))


def get_completed_timestamp_from_run_id(
        conn: ConnectionPlus, run_id: int) -> float:
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
    return select_one_where(conn, "runs", "completed_timestamp",
                            "run_id", run_id)


def get_guid_from_run_id(conn: ConnectionPlus, run_id: int) -> str:
    """
    Get the guid of the given run

    Args:
        conn: database connection
        run_id: id of the run
    """
    return select_one_where(conn, "runs", "guid", "run_id", run_id)


def finish_experiment(conn: ConnectionPlus, exp_id: int) -> None:
    """ Finish experiment

    Args:
        conn: database connection
        exp_id: the id of the experiment
    """
    query = """
    UPDATE experiments SET end_time=? WHERE exp_id=?;
    """
    atomic_transaction(conn, query, time.time(), exp_id)


def get_run_counter(conn: ConnectionPlus, exp_id: int) -> int:
    """ Get the experiment run counter

    Args:
        conn: the connection to the sqlite database
        exp_id: experiment identifier

    Returns:
        the experiment run counter

    """
    return select_one_where(conn, "experiments", "run_counter",
                            where_column="exp_id",
                            where_value=exp_id)


def get_experiments(conn: ConnectionPlus) -> List[sqlite3.Row]:
    """ Get a list of experiments
     Args:
         conn: database connection

     Returns:
         list of rows
     """
    sql = """
    SELECT * FROM experiments
    """
    c = atomic_transaction(conn, sql)

    return c.fetchall()


def get_matching_exp_ids(conn: ConnectionPlus,
                         **match_conditions: Any) -> List[int]:
    """
    Get exp_ids for experiments matching the match_conditions

    Raises:
        ValueError if a match_condition that is not "name", "sample_name",
        "format_string", "run_counter", "start_time", or "end_time"
    """
    valid_conditions = ["name", "sample_name", "start_time", "end_time",
                        "run_counter", "format_string"]

    for mcond in match_conditions:
        if mcond not in valid_conditions:
            raise ValueError(f"{mcond} is not a valid match condition.")

    end_time = match_conditions.get('end_time', None)
    time_eq = "=" if end_time is not None else "IS"

    sample_name = match_conditions.get('sample_name', None)
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
        query = query.replace("format_string = ?",
                              f'format_string = "{format_string}"')
        match_conditions.pop("format_string")
    query = query.replace("end_time = ?", f"end_time {time_eq} ?")
    query = query.replace("sample_name = ?", f"sample_name {sample_name_eq} ?")

    cursor = conn.cursor()
    cursor.execute(query, tuple(match_conditions.values()))
    rows = cursor.fetchall()

    return [row[0] for row in rows]


def get_exp_ids_from_run_ids(conn: ConnectionPlus,
                             run_ids: Sequence[int]) -> List[int]:
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


def get_last_experiment(conn: ConnectionPlus) -> Optional[int]:
    """
    Return last started experiment id

    Returns None if there are no experiments in the database
    """
    query = "SELECT MAX(exp_id) FROM experiments"
    c = atomic_transaction(conn, query)
    return c.fetchall()[0][0]


def get_runs(conn: ConnectionPlus,
             exp_id: Optional[int] = None) -> List[sqlite3.Row]:
    """ Get a list of runs.

    Args:
        conn: database connection
        exp_id: id of the experiment to look inside.
            If None all experiments will be included

    Returns:
        list of rows
    """
    with atomic(conn) as conn:
        if exp_id:
            sql = """
            SELECT * FROM runs
            where exp_id = ?
            """
            c = transaction(conn, sql, exp_id)
        else:
            sql = """
            SELECT * FROM runs
            """
            c = transaction(conn, sql)

    return c.fetchall()


def get_last_run(conn: ConnectionPlus,
                 exp_id: Optional[int] = None) -> Optional[int]:
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
    return one(c, 'run_id')


def run_exists(conn: ConnectionPlus, run_id: int) -> bool:
    # the following query always returns a single sqlite3.Row with an integer
    # value of `1` or `0` for existing and non-existing run_id in the database
    query = """
    SELECT EXISTS(
        SELECT 1
        FROM runs
        WHERE run_id = ?
        LIMIT 1
    );
    """
    res: sqlite3.Row = atomic_transaction(conn, query, run_id).fetchone()
    return bool(res[0])


def data_sets(conn: ConnectionPlus) -> List[sqlite3.Row]:
    """ Get a list of datasets
    Args:
        conn: database connection

    Returns:
        list of rows
    """
    sql = """
    SELECT * FROM runs
    """
    c = atomic_transaction(conn, sql)
    return c.fetchall()


def format_table_name(fmt_str: str, name: str, exp_id: int,
                      run_counter: int) -> str:
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


def _insert_run(conn: ConnectionPlus, exp_id: int, name: str,
                guid: str,
                parameters: Optional[List[ParamSpec]] = None,
                captured_run_id: Optional[int] = None,
                captured_counter: Optional[int] = None,
                parent_dataset_links: str = "[]"
                ) -> Tuple[int, str, int]:

    # get run counter and formatter from experiments
    run_counter, format_string = select_many_where(conn,
                                                   "experiments",
                                                   "run_counter",
                                                   "format_string",
                                                   where_column="exp_id",
                                                   where_value=exp_id)
    run_counter += 1
    if captured_counter is None:
        with atomic(conn) as conn:
            query = """
            SELECT
                max(captured_counter)
            FROM
                runs
            WHERE
                exp_id = ?"""
            curr = transaction(conn, query, exp_id)
            existing_captured_counter = one(curr, 0)
            if existing_captured_counter is not None:
                captured_counter = existing_captured_counter + 1
            else:
                captured_counter = run_counter
    formatted_name = format_table_name(format_string, name, exp_id,
                                       run_counter)
    table = "runs"

    parameters = parameters or []

    run_desc = RunDescriber(old_to_new(v0.InterDependencies(*parameters)))
    desc_str = serial.to_json_for_storage(run_desc)

    if captured_run_id is None:
        with atomic(conn) as conn:
            query = """
            SELECT
                max(captured_run_id)
            FROM
                runs"""
            curr = transaction(conn, query)
            existing_captured_run_id = one(curr, 0)
        if existing_captured_run_id is not None:
            captured_run_id = existing_captured_run_id + 1
        else:
            captured_run_id = 1

    with atomic(conn) as conn:

        if parameters:
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
            curr = transaction(conn, query,
                               name,
                               exp_id,
                               guid,
                               formatted_name,
                               run_counter,
                               None,
                               ",".join([p.name for p in parameters]),
                               False,
                               desc_str,
                               captured_run_id,
                               captured_counter,
                               parent_dataset_links)

            _add_parameters_to_layout_and_deps(conn, formatted_name,
                                               *parameters)

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
            curr = transaction(conn, query,
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
                               parent_dataset_links)

    run_id = curr.lastrowid

    return run_counter, formatted_name, run_id


def _update_experiment_run_counter(conn: ConnectionPlus, exp_id: int,
                                   run_counter: int) -> None:
    query = """
    UPDATE experiments
    SET run_counter = ?
    WHERE exp_id = ?
    """
    atomic_transaction(conn, query, run_counter, exp_id)


def _get_parameters(conn: ConnectionPlus,
                    run_id: int) -> List[ParamSpec]:
    """
    Get the list of param specs for run

    Args:
        conn: the connection to the sqlite database
        run_id: The id of the run

    Returns:
        A list of param specs for this run
    """

    sql = f"""
    SELECT parameter FROM layouts WHERE run_id={run_id}
    """
    c = conn.execute(sql)
    param_names_temp = many_many(c, 'parameter')
    param_names = [p[0] for p in param_names_temp]
    param_names = cast(List[str], param_names)

    parspecs = []

    for param_name in param_names:
        parspecs.append(_get_paramspec(conn, run_id, param_name))

    return parspecs


def _get_paramspec(conn: ConnectionPlus,
                   run_id: int,
                   param_name: str) -> ParamSpec:
    """
    Get the ParamSpec object for the given parameter name
    in the given run

    Args:
        conn: Connection to the database
        run_id: The run id
        param_name: The name of the parameter
    """

    # get table name
    sql = f"""
    SELECT result_table_name FROM runs WHERE run_id = {run_id}
    """
    c = conn.execute(sql)
    result_table_name = one(c, 'result_table_name')

    # get the data type
    sql = f"""
    PRAGMA TABLE_INFO("{result_table_name}")
    """
    c = conn.execute(sql)
    for row in c.fetchall():
        if row['name'] == param_name:
            param_type = row['type']
            break

    # get everything else

    sql = f"""
    SELECT * FROM layouts
    WHERE parameter="{param_name}" and run_id={run_id}
    """
    c = conn.execute(sql)
    resp = many(c, 'layout_id', 'run_id', 'parameter', 'label', 'unit',
                'inferred_from')
    (layout_id, _, _, label, unit, inferred_from_string) = resp

    if inferred_from_string:
        inferred_from = inferred_from_string.split(', ')
    else:
        inferred_from = []

    deps = _get_dependencies(conn, layout_id)
    depends_on: Optional[List[str]]
    if len(deps) == 0:
        depends_on = None
    else:
        dps: List[int] = [dp[0] for dp in deps]
        ax_nums: List[int] = [dp[1] for dp in deps]
        depends_on = []
        for _, dp in sorted(zip(ax_nums, dps)):
            sql = f"""
            SELECT parameter FROM layouts WHERE layout_id = {dp}
            """
            c = conn.execute(sql)
            depends_on.append(one(c, 'parameter'))

    parspec = ParamSpec(param_name, param_type, label, unit,
                        inferred_from,
                        depends_on)
    return parspec


def update_run_description(conn: ConnectionPlus, run_id: int,
                           description: str) -> None:
    """
    Update the run_description field for the given run_id. The description
    string must be a valid JSON string representation of a RunDescriber object
    """
    try:
        serial.from_json_to_current(description)
    except Exception as e:
        raise ValueError("Invalid description string. Must be a JSON string "
                         "representation of a RunDescriber object.") from e

    _update_run_description(conn, run_id, description)


def _update_run_description(conn: ConnectionPlus, run_id: int,
                            description: str) -> None:
    """
    Update the run_description field for the given run_id. The description
    string is NOT validated.
    """
    sql = """
          UPDATE runs
          SET run_description = ?
          WHERE run_id = ?
          """
    with atomic(conn) as conn:
        conn.cursor().execute(sql, (description, run_id))


def update_parent_datasets(conn: ConnectionPlus,
                           run_id: int, links_str: str) -> None:
    """
    Update (i.e. overwrite) the parent_datasets field for the given run_id
    """
    if not is_column_in_table(conn, 'runs', 'parent_datasets'):
        insert_column(conn, 'runs', 'parent_datasets')

    sql = """
          UPDATE runs
          SET parent_datasets = ?
          WHERE run_id = ?
          """
    with atomic(conn) as conn:
        conn.cursor().execute(sql, (links_str, run_id))


def set_run_timestamp(conn: ConnectionPlus, run_id: int) -> None:
    """
    Set the run_timestamp for the run with the given run_id. If the
    run_timestamp has already been set, a RuntimeError is raised.
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

    with atomic(conn) as conn:
        c = conn.cursor()
        timestamp = one(c.execute(query, (run_id,)), 'run_timestamp')
        if timestamp is not None:
            raise RuntimeError('Can not set run_timestamp; it has already '
                               f'been set to: {timestamp}')
        else:
            current_time = time.time()
            c.execute(cmd, (current_time, run_id))
            log.info(f"Set the run_timestamp of run_id {run_id} to "
                     f"{current_time}")


def add_parameter(conn: ConnectionPlus,
                  formatted_name: str,
                  *parameter: ParamSpec) -> None:
    """
    Add parameters to the dataset

    This will update the layouts and dependencies tables

    NOTE: two parameters with the same name are not allowed

    Args:
        conn: the connection to the sqlite database
        formatted_name: name of the table
        parameter: the list of ParamSpecs for parameters to add
    """
    with atomic(conn) as conn:
        p_names = []
        for p in parameter:
            insert_column(conn, formatted_name, p.name, p.type)
            p_names.append(p.name)
        # get old parameters column from run table
        sql = f"""
        SELECT parameters FROM runs
        WHERE result_table_name=?
        """
        with atomic(conn) as conn:
            c = transaction(conn, sql, formatted_name)
        old_parameters = one(c, 'parameters')
        if old_parameters:
            new_parameters = ",".join([old_parameters] + p_names)
        else:
            new_parameters = ",".join(p_names)
        sql = "UPDATE runs SET parameters=? WHERE result_table_name=?"
        with atomic(conn) as conn:
            transaction(conn, sql, new_parameters, formatted_name)

        # Update the layouts table
        c = _add_parameters_to_layout_and_deps(conn, formatted_name,
                                               *parameter)


def _add_parameters_to_layout_and_deps(conn: ConnectionPlus,
                                       formatted_name: str,
                                       *parameter: ParamSpec
                                       ) -> sqlite3.Cursor:
    # get the run_id
    sql = f"""
    SELECT run_id FROM runs WHERE result_table_name="{formatted_name}";
    """
    run_id = one(transaction(conn, sql), 'run_id')
    layout_args = []
    for p in parameter:
        layout_args.append(run_id)
        layout_args.append(p.name)
        layout_args.append(p.label)
        layout_args.append(p.unit)
        layout_args.append(p.inferred_from)
    rowplaceholder = '(?, ?, ?, ?, ?)'
    placeholder = ','.join([rowplaceholder] * len(parameter))
    sql = f"""
    INSERT INTO layouts (run_id, parameter, label, unit, inferred_from)
    VALUES {placeholder}
    """

    with atomic(conn) as conn:
        c = transaction(conn, sql, *layout_args)

        for p in parameter:

            if p.depends_on != '':

                layout_id = _get_layout_id(conn, p, run_id)

                deps = p.depends_on.split(', ')
                for ax_num, dp in enumerate(deps):

                    sql = """
                    SELECT layout_id FROM layouts
                    WHERE run_id=? and parameter=?;
                    """

                    c = transaction(conn, sql, run_id, dp)
                    dep_ind = one(c, 'layout_id')

                    sql = """
                    INSERT INTO dependencies (dependent, independent, axis_num)
                    VALUES (?,?,?)
                    """

                    c = transaction(conn, sql, layout_id, dep_ind, ax_num)
    return c


def _validate_table_name(table_name: str) -> bool:
    valid = True
    for i in table_name:
        if unicodedata.category(i) not in _unicode_categories:
            valid = False
            raise RuntimeError("Invalid table name "
                               "{} starting at {}".format(table_name, i))
    return valid


def _create_run_table(conn: ConnectionPlus,
                      formatted_name: str,
                      parameters: Optional[List[ParamSpec]] = None,
                      values: Optional[VALUES] = None
                      ) -> None:
    """Create run table with formatted_name as name

    Args:
        conn: database connection
        formatted_name: the name of the table to create
    """
    _validate_table_name(formatted_name)

    with atomic(conn) as conn:

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
            insert_values(conn, formatted_name,
                          [p.name for p in parameters], values)
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
            query = f"""
            CREATE TABLE "{formatted_name}" (
                id INTEGER PRIMARY KEY
            );
            """
            transaction(conn, query)


def create_run(conn: ConnectionPlus, exp_id: int, name: str,
               guid: str,
               parameters: Optional[List[ParamSpec]] = None,
               values:  Optional[List[Any]] = None,
               metadata: Optional[Mapping[str, Any]] = None,
               captured_run_id: Optional[int] = None,
               captured_counter: Optional[int] = None,
               parent_dataset_links: str = "[]"
               ) -> Tuple[int, int, str]:
    """ Create a single run for the experiment.


    This will register the run in the runs table, the counter in the
    experiments table and create a new table with the formatted name.

    Args:
        - conn: the connection to the sqlite database
        - exp_id: the experiment id we want to create the run into
        - name: a friendly name for this run
        - guid: the guid adhering to our internal guid format
        - parameters: optional list of parameters this run has
        - values:  optional list of values for the parameters
        - metadata: optional metadata dictionary
        - captured_run_id: The run_id this data was originally captured with.
            Should only be supplied when inserting an already completed run
            from another database into this database. Otherwise leave as None.
        - captured_counter: The counter this data was originally captured with.
            Should only be supplied when inserting an already completed run
            from another database into this database. Otherwise leave as None.

    Returns:
        - run_counter: the id of the newly created run (not unique)
        - run_id: the row id of the newly created run
        - formatted_name: the name of the newly created table
    """

    with atomic(conn):
        run_counter, formatted_name, run_id = _insert_run(conn,
                                                          exp_id,
                                                          name,
                                                          guid,
                                                          parameters,
                                                          captured_run_id,
                                                          captured_counter,
                                                          parent_dataset_links)
        if metadata:
            add_meta_data(conn, run_id, metadata)
        _update_experiment_run_counter(conn, exp_id, run_counter)
        _create_run_table(conn, formatted_name, parameters, values)

    return run_counter, run_id, formatted_name


def get_run_description(conn: ConnectionPlus, run_id: int) -> str:
    """
    Return the (JSON string) run description of the specified run
    """
    return select_one_where(conn, "runs", "run_description",
                            "run_id", run_id)


def get_parent_dataset_links(conn: ConnectionPlus, run_id: int) -> str:
    """
    Return the (JSON string) of the parent-child dataset links for the
    specified run
    """

    # We cannot in general trust that NULLs will not appear in the column,
    # even if the column is present in the runs table.

    link_str: str
    maybe_link_str: Optional[str]

    if not is_column_in_table(conn, 'runs', 'parent_datasets'):
        maybe_link_str = None
    else:
        maybe_link_str = select_one_where(conn, "runs", "parent_datasets",
                                           "run_id", run_id)

    if maybe_link_str is None:
        link_str = "[]"
    else:
        link_str = str(maybe_link_str)

    return link_str


def get_metadata(conn: ConnectionPlus, tag: str, table_name: str) -> str:
    """ Get metadata under the tag from table
    """
    return select_one_where(conn, "runs", tag,
                            "result_table_name", table_name)


def get_metadata_from_run_id(conn: ConnectionPlus, run_id: int) -> Dict:
    """
    Get all metadata associated with the specified run
    """
    non_metadata = RUNS_TABLE_COLUMNS

    metadata = {}
    possible_tags = []

    # first fetch all columns of the runs table
    query = "PRAGMA table_info(runs)"
    cursor = conn.cursor()
    for row in cursor.execute(query):
        if row['name'] not in non_metadata:
            possible_tags.append(row['name'])

    # and then fetch whatever metadata the run might have
    for tag in possible_tags:
        query = f"""
                SELECT "{tag}"
                FROM runs
                WHERE run_id = ?
                AND "{tag}" IS NOT NULL
                """
        cursor.execute(query, (run_id,))
        row = cursor.fetchall()
        if row != []:
            metadata[tag] = row[0][tag]

    return metadata


def insert_meta_data(conn: ConnectionPlus, row_id: int, table_name: str,
                     metadata: Mapping[str, Any]) -> None:
    """
    Insert new metadata column and add values. Note that None is not a valid
    metadata value

    Args:
        - conn: the connection to the sqlite database
        - row_id: the row to add the metadata at
        - table_name: the table to add to, defaults to runs
        - metadata: the metadata to add
    """
    for tag, val in metadata.items():
        if val is None:
            raise ValueError(f'Tag {tag} has value None. '
                             ' That is not a valid metadata value!')
    for key in metadata.keys():
        insert_column(conn, table_name, key)
    update_meta_data(conn, row_id, table_name, metadata)


def update_meta_data(conn: ConnectionPlus, row_id: int, table_name: str,
                     metadata: Mapping[str, Any]) -> None:
    """
    Updates metadata (they must exist already)

    Args:
        - conn: the connection to the sqlite database
        - row_id: the row to add the metadata at
        - table_name: the table to add to, defaults to runs
        - metadata: the metadata to add
    """
    update_where(conn, table_name, 'rowid', row_id, **metadata)


def add_meta_data(conn: ConnectionPlus,
                  row_id: int,
                  metadata: Mapping[str, Any],
                  table_name: str = "runs") -> None:
    """
    Add metadata data (updates if exists, create otherwise).
    Note that None is not a valid metadata value.

    Args:
        - conn: the connection to the sqlite database
        - row_id: the row to add the metadata at
        - metadata: the metadata to add
        - table_name: the table to add to, defaults to runs
    """
    try:
        insert_meta_data(conn, row_id, table_name, metadata)
    except sqlite3.OperationalError as e:
        # this means that the column already exists
        # so just insert the new value
        if str(e).startswith("duplicate"):
            update_meta_data(conn, row_id, table_name, metadata)
        else:
            raise e


def get_experiment_name_from_experiment_id(
        conn: ConnectionPlus, exp_id: int) -> str:
    return select_one_where(
        conn, "experiments", "name", "exp_id", exp_id)


def get_sample_name_from_experiment_id(
        conn: ConnectionPlus, exp_id: int) -> str:
    return select_one_where(
        conn, "experiments", "sample_name", "exp_id", exp_id)


def get_run_timestamp_from_run_id(conn: ConnectionPlus,
                                  run_id: int) -> Optional[float]:
    return select_one_where(conn, "runs", "run_timestamp", "run_id", run_id)


def update_GUIDs(conn: ConnectionPlus) -> None:
    """
    Update all GUIDs in this database where either the location code or the
    work_station code is zero to use the location and work_station code from
    the qcodesrc.json file in home. Runs where it is not true that both codes
    are zero are skipped.
    """

    log.info('Commencing update of all GUIDs in database')

    cfg = qc.config

    location = cfg['GUID_components']['location']
    work_station = cfg['GUID_components']['work_station']

    if location == 0:
        log.warning('The location is still set to the default (0). Can not '
                    'proceed. Please configure the location before updating '
                    'the GUIDs.')
        return
    if work_station == 0:
        log.warning('The work_station is still set to the default (0). Can not'
                    ' proceed. Please configure the location before updating '
                    'the GUIDs.')
        return

    query = f"select MAX(run_id) from runs"
    c = atomic_transaction(conn, query)
    no_of_runs = c.fetchall()[0][0]

    # now, there are four actions we can take

    def _both_nonzero(run_id: int, *args: Any) -> None:
        log.info(f'Run number {run_id} already has a valid GUID, skipping.')

    def _location_only_zero(run_id: int, *args: Any) -> None:
        log.warning(f'Run number {run_id} has a zero (default) location '
                    'code, but a non-zero work station code. Please manually '
                    'resolve this, skipping the run now.')

    def _workstation_only_zero(run_id: int, *args: Any) -> None:
        log.warning(f'Run number {run_id} has a zero (default) work station'
                    ' code, but a non-zero location code. Please manually '
                    'resolve this, skipping the run now.')

    def _both_zero(run_id: int,
                   conn: ConnectionPlus,
                   guid_comps: Dict[str, Any]) -> None:
        guid_str = generate_guid(timeint=guid_comps['time'],
                                 sampleint=guid_comps['sample'])
        with atomic(conn) as conn:
            sql = f"""
                   UPDATE runs
                   SET guid = ?
                   where run_id == {run_id}
                   """
            cur = conn.cursor()
            cur.execute(sql, (guid_str,))

        log.info(f'Succesfully updated run number {run_id}.')

    actions: Dict[Tuple[bool, bool], Callable]
    actions = {(True, True): _both_zero,
               (False, True): _workstation_only_zero,
               (True, False): _location_only_zero,
               (False, False): _both_nonzero}

    for run_id in range(1, no_of_runs+1):
        guid_str = get_guid_from_run_id(conn, run_id)
        guid_comps = parse_guid(guid_str)
        loc = guid_comps['location']
        ws = guid_comps['work_station']

        log.info(f'Updating run number {run_id}...')
        actions[(loc == 0, ws == 0)](run_id, conn, guid_comps)


def remove_trigger(conn: ConnectionPlus, trigger_id: str) -> None:
    """
    Removes a trigger with a given id if it exists.

    Note that this transaction is not atomic!

    Args:
        conn: database connection object
        trigger_id: id of the trigger
    """
    transaction(conn, f"DROP TRIGGER IF EXISTS {trigger_id};")


def append_shaped_parameter_data_to_existing_arrays(
        conn: ConnectionPlus,
        table_name: str,
        rundescriber: RunDescriber,
        write_status: Dict[str, Optional[int]],
        read_status: Dict[str, int],
        data: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[Dict[str, Optional[int]],
           Dict[str, int],
           Dict[str, Dict[str, np.ndarray]]]:
    """
    Append newly loaded data to an already existing cache.

    Args:
        conn: The connection to the sqlite database
        table_name: The name of the table the data is stored in
        rundescriber: The rundescriber that describes the run
        write_status: Mapping from dependent parameter name to number of rows
          written to the cache previously.
        read_status: Mapping from dependent parameter name to number of rows
          read from the db previously.
        data: Mapping from dependent parameter name to mapping
          from parameter name to numpy arrays that the data should be
          inserted into.

    Returns:
        Updated write and read status, and the updated ``data``
    """
    parameters = tuple(ps.name for ps in
                       rundescriber.interdeps.non_dependencies)
    merged_data = {}

    updated_write_status = copy(write_status)
    updated_read_status = copy(read_status)

    for meas_parameter in parameters:

        shapes = rundescriber.shapes
        if shapes is not None:
            shape = shapes.get(meas_parameter, None)
        else:
            shape = None

        start = read_status.get(meas_parameter, 0) + 1

        new_data, n_rows_read = get_parameter_data_for_one_paramtree(
            conn,
            table_name,
            rundescriber=rundescriber,
            output_param=meas_parameter,
            start=start,
            end=None
        )

        existing_data = data.get(meas_parameter, {})

        subtree_merged_data = {}
        subtree_parameters = set(existing_data.keys()) | set(new_data.keys())
        new_write_status: Optional[int]

        for subtree_param in subtree_parameters:
            existing_values = existing_data.get(subtree_param)
            new_values = new_data.get(subtree_param)
            if existing_values is not None and new_values is not None:
                (subtree_merged_data[subtree_param],
                 new_write_status) = _insert_into_data_dict(
                    existing_values,
                    new_values,
                    write_status.get(meas_parameter),
                    shape=shape
                )
                updated_write_status[meas_parameter] = new_write_status
            elif new_values is not None:
                (subtree_merged_data[subtree_param],
                 new_write_status) = _create_new_data_dict(
                    new_values,
                    shape
                )
                updated_write_status[meas_parameter] = new_write_status
            elif existing_values is not None:
                subtree_merged_data[subtree_param] = existing_values
        merged_data[meas_parameter] = subtree_merged_data
        updated_read_status[meas_parameter] = read_status.get(meas_parameter, 0) + n_rows_read
    return updated_write_status, updated_read_status, merged_data


def _create_new_data_dict(new_values: np.ndarray,
                          shape: Optional[Tuple[int, ...]]
                          ) -> Tuple[np.ndarray, int]:
    if shape is None:
        return new_values, new_values.size
    else:
        n_values = new_values.size
        data = np.zeros(shape, dtype=new_values.dtype)

        if new_values.dtype.kind == "f" or new_values.dtype.kind == "c":
            data[:] = np.nan

        data.ravel()[0:n_values] = new_values
        return data, n_values


def _insert_into_data_dict(
        existing_values: np.ndarray,
        new_values: np.ndarray,
        write_status: Optional[int],
        shape: Optional[Tuple[int, ...]]
) -> Tuple[np.ndarray, Optional[int]]:
    if shape is None or write_status is None:
        return np.append(existing_values, new_values, axis=0), None
    else:
        if existing_values.dtype.kind in ('U', 'S'):
            # string type arrays may be too small for the new data
            # read so rescale if needed.
            if new_values.dtype.itemsize > existing_values.dtype.itemsize:
                existing_values = existing_values.astype(new_values.dtype)
        n_values = new_values.size
        new_write_status = write_status+n_values
        if new_write_status > existing_values.size:
            log.warning(f"Incorrect shape of dataset: Dataset is expected to "
                        f"contain {existing_values.size} points but trying to "
                        f"add an amount of data that makes it contain {new_write_status} points. Cache will "
                        f"be flattened into a 1D array")
            return (np.append(existing_values.flatten(),
                              new_values.flatten(), axis=0),
                    new_write_status)
        else:
            existing_values.ravel()[write_status:new_write_status] = new_values
            return existing_values, new_write_status
