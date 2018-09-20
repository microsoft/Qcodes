#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 unga <giulioungaretti@me.com>
#
# Distributed under terms of the MIT license.
# import json
import functools
from typing import Any, Dict, List, Optional, Union, Sized, Callable
from threading import Thread
import time
import logging
import hashlib
import uuid
from queue import Queue, Empty
import warnings

from qcodes.dataset.param_spec import ParamSpec
from qcodes.instrument.parameter import _BaseParameter
from qcodes.dataset.sqlite_base import (atomic, atomic_transaction,
                                        transaction, add_parameter,
                                        connect, create_run, completed,
                                        get_parameters,
                                        get_experiments,
                                        get_last_experiment, select_one_where,
                                        length, modify_values,
                                        add_meta_data, mark_run_complete,
                                        modify_many_values, insert_values,
                                        insert_many_values,
                                        VALUE, VALUES, get_data,
                                        get_values,
                                        get_setpoints,
                                        get_metadata, one,
                                        get_experiment_name_from_experiment_id,
                                        get_sample_name_from_experiment_id,
                                        get_guid_from_run_id,
                                        get_run_timestamp_from_run_id,
                                        get_completed_timestamp_from_run_id)
from qcodes.dataset.database import get_DB_location
from qcodes.dataset.guids import generate_guid

# TODO: as of now every time a result is inserted with add_result the db is
# saved same for add_results. IS THIS THE BEHAVIOUR WE WANT?

# TODO: storing parameters in separate table as an extension (dropping
# the column parametenrs would be much nicer

# TODO: metadata split between well known columns and maybe something else is
# not such a good idea. The problem is if we allow for specific columns then
# how do the user/us know which are metatadata?  I THINK the only sane solution
# is to store JSON in a column called metadata

# TODO: we cant have parameters with the same name in the same dataset/run

# TODO: fixix  a subset of metadata that we define well known (and create them)
# i.e. no dynamic creation of metadata columns, but add stuff to
# a json inside a 'metadata' column


# SPECS is a list of ParamSpec
SPECS = List[ParamSpec]


class CompletedError(RuntimeError):
    pass


class _Subscriber(Thread):
    """
    Class to add a subscriber to a DataSet. The subscriber gets called
    every time an insert is made to the results_table.

    The _Subscriber is not meant to be instantiated directly, but rather used
    via the 'subscribe' method of the DataSet.

    NOTE: A subscriber should be added *after* all parameters have been added.

    NOTE: Special care shall be taken when using the *state* object: it is the user's
    responsibility to operate with it in a thread-safe way.
    """
    def __init__(self,
                 dataSet,
                 id_: str,
                 callback: Callable[..., None],
                 state: Optional[Any] = None,
                 loop_sleep_time: int = 0,  # in milliseconds
                 min_queue_length: int = 1,
                 callback_kwargs: Optional[Dict[str, Any]]=None
                 ) -> None:
        super().__init__()

        self._id = id_

        self.dataSet = dataSet
        self.table_name = dataSet.table_name
        self._data_set_len = len(dataSet)

        self.state = state

        self.data_queue: Queue = Queue()
        self._queue_length: int = 0
        self._stop_signal: bool = False
        self._loop_sleep_time = loop_sleep_time / 1000  # convert milliseconds to seconds
        self.min_queue_length = min_queue_length

        if callback_kwargs is None:
            self.callback = callback
        else:
            self.callback = functools.partial(callback, **callback_kwargs)

        self.callback_id = f"callback{self._id}"

        conn = dataSet.conn

        conn.create_function(self.callback_id, -1, self._cache_data_to_queue)

        parameters = dataSet.get_parameters()
        sql_param_list = ",".join([f"NEW.{p.name}" for p in parameters])
        sql_create_trigger_for_callback = f"""
        CREATE TRIGGER sub{self._id}
            AFTER INSERT ON '{self.table_name}'
        BEGIN
            SELECT {self.callback_id}({sql_param_list});
        END;"""
        atomic_transaction(conn, sql_create_trigger_for_callback)

        self.log = logging.getLogger(f"_Subscriber {self._id}")

    def _cache_data_to_queue(self, *args) -> None:
        self.log.debug(f"Args:{args} put into queue for {self.callback_id}")
        self.data_queue.put(args)
        self._data_set_len += 1
        self._queue_length += 1

    def run(self) -> None:
        self.log.debug("Starting subscriber")
        self._loop()

    @staticmethod
    def _exhaust_queue(queue: Queue) -> List:
        result_list = []
        while True:
            try:
                result_list.append(queue.get(block=False))
            except Empty:
                break
        return result_list

    def _call_callback_on_queue_data(self) -> None:
        result_list = self._exhaust_queue(self.data_queue)
        self.callback(result_list, self._data_set_len, self.state)
        self.log.debug(f"{self.callback} called with "
                       f"result_list: {result_list}.")

    def _loop(self) -> None:
        while True:
            if self._stop_signal:
                self._clean_up()
                break

            if self._queue_length >= self.min_queue_length:
                self._call_callback_on_queue_data()
                self._queue_length = 0

            time.sleep(self._loop_sleep_time)

            if self.dataSet.completed:
                self._call_callback_on_queue_data()
                break

    def done_callback(self) -> None:
        self.log.debug("Done callback")
        self._call_callback_on_queue_data()

    def schedule_stop(self) -> None:
        if not self._stop_signal:
            self.log.debug("Scheduling stop")
            self._stop_signal = True

    def _clean_up(self) -> None:
        self.log.debug("Stopped subscriber")


class DataSet(Sized):
    def __init__(self, path_to_db: str, run_id: Optional[int]=None,
                 conn=None) -> None:
        """
        Create a new DataSet object. The object can either be intended
        to hold a new run or and old run.

        Args:
            path_to_db: path to the sqlite file on disk
            run_id: provide this when loading an existing run, leave it
              as None when creating a new run
            conn: connection to the DB
        """
        # TODO: handle fail here by defaulting to
        # a standard db
        self.path_to_db = path_to_db
        if conn is None:
            self.conn = connect(self.path_to_db)
        else:
            self.conn = conn

        self.run_id = run_id
        self._debug = False
        self.subscribers: Dict[str, _Subscriber] = {}
        if run_id:
            self._completed = completed(self.conn, self.run_id)

    def _new(self, name, exp_id, specs: SPECS = None, values=None,
             metadata=None) -> None:
        """
        Actually perform all the side effects needed for
        the creation of a new dataset.
        """
        _, run_id, __ = create_run(self.conn, exp_id, name,
                                   generate_guid(),
                                   specs, values, metadata)

        # this is really the UUID (an ever increasing count in the db)
        self.run_id = run_id
        self._completed = False

    @property
    def name(self):
        return select_one_where(self.conn, "runs",
                                "name", "run_id", self.run_id)

    @property
    def table_name(self):
        return select_one_where(self.conn, "runs",
                                "result_table_name", "run_id", self.run_id)

    @property
    def guid(self):
        return get_guid_from_run_id(self.conn, self.run_id)

    @property
    def number_of_results(self):
        tabnam = self.table_name
        # TODO: is it better/faster to use the max index?
        sql = f'SELECT COUNT(*) FROM "{tabnam}"'
        cursor = atomic_transaction(self.conn, sql)
        return one(cursor, 'COUNT(*)')

    @property
    def counter(self):
        return select_one_where(self.conn, "runs",
                                "result_counter", "run_id", self.run_id)

    @property
    def parameters(self) -> str:
        return select_one_where(self.conn, "runs",
                                "parameters", "run_id", self.run_id)

    @property
    def paramspecs(self) -> Dict[str, ParamSpec]:
        params = self.get_parameters()
        param_names = [p.name for p in params]
        return dict(zip(param_names, params))

    @property
    def exp_id(self) -> int:
        return select_one_where(self.conn, "runs",
                                "exp_id", "run_id", self.run_id)

    @property
    def exp_name(self) -> str:
        return get_experiment_name_from_experiment_id(self.conn, self.exp_id)

    @property
    def sample_name(self) -> str:
        return get_sample_name_from_experiment_id(self.conn, self.exp_id)

    @property
    def run_timestamp_raw(self) -> float:
        """
        Returns run timestamp as number of seconds since the Epoch

        The run timestamp is the moment when the measurement for this run
        started.
        """
        return get_run_timestamp_from_run_id(self.conn, self.run_id)

    def run_timestamp(self, fmt: str="%Y-%m-%d %H:%M:%S") -> str:
        """
        Returns run timestamp in a human-readable format

        The run timestamp is the moment when the measurement for this run
        started.

        Consult with `time.strftime` for information about the format.
        """
        return time.strftime(fmt, time.localtime(self.run_timestamp_raw))

    @property
    def completed_timestamp_raw(self) -> Union[float, None]:
        """
        Returns timestamp when measurement run was completed
        as number of seconds since the Epoch

        If the run (or the dataset) is not completed, then returns None.
        """
        return get_completed_timestamp_from_run_id(self.conn, self.run_id)

    def completed_timestamp(self,
                            fmt: str="%Y-%m-%d %H:%M:%S") -> Union[str, None]:
        """
        Returns timestamp when measurement run was completed
        in a human-readable format

        If the run (or the dataset) is not completed, then returns None.

        Consult with `time.strftime` for information about the format.
        """
        completed_timestamp_raw = self.completed_timestamp_raw

        if completed_timestamp_raw:
            completed_timestamp = time.strftime(
                fmt, time.localtime(completed_timestamp_raw))
        else:
            completed_timestamp = None

        return completed_timestamp

    def toggle_debug(self):
        """
        Toggle debug mode, if debug mode is on
        all the queries made are echoed back.
        """
        self._debug = not self._debug
        self.conn.close()
        self.conn = connect(self.path_to_db, self._debug)

    def add_parameter(self, spec: ParamSpec):
        """
        Add a parameter to the DataSet. To ensure sanity, parameters must be
        added to the DataSet in a sequence matching their internal
        dependencies, i.e. first independent parameters, next other
        independent parameters inferred from the first ones, and finally
        the dependent parameters
        """
        if self.parameters:
            old_params = self.parameters.split(',')
        else:
            old_params = []

        # better to catch this one early
        # alternatively make this a warning and a NOOP
        if spec.name in old_params:
            raise ValueError(f'Duplicate parameter name: {spec.name}')

        inf_from = spec.inferred_from.split(', ')
        if inf_from == ['']:
            inf_from = []
        for ifrm in inf_from:
            if ifrm not in old_params:
                raise ValueError('Can not infer parameter '
                                 f'{spec.name} from {ifrm}, '
                                 'no such parameter in this DataSet')

        dep_on = spec.depends_on.split(', ')
        if dep_on == ['']:
            dep_on = []
        for dp in dep_on:
            if dp not in old_params:
                raise ValueError('Can not have parameter '
                                 f'{spec.name} depend on {dp}, '
                                 'no such parameter in this DataSet')

        add_parameter(self.conn, self.table_name, spec)

    def get_parameters(self) -> SPECS:
        return get_parameters(self.conn, self.run_id)

    # TODO: deprecate
    def add_parameters(self, specs: SPECS):
        add_parameter(self.conn, self.table_name, *specs)

    def add_metadata(self, tag: str, metadata: Any):
        """
        Adds metadata to the DataSet. The metadata is stored under the
        provided tag.

        Args:
            tag: represents the key in the metadata dictionary
            metadata: actual metadata

        """
        # TODO: this follows the spec but another option:
        # json_meta_data = json.dumps(metadata)
        # add_meta_data(self.conn, self.run_id, {"metadata": json_meta_data})

        add_meta_data(self.conn, self.run_id, {tag: metadata})
        # adding meta-data does not commit
        self.conn.commit()

    @property
    def completed(self) -> bool:
        return self._completed

    @completed.setter
    def completed(self, value):
        self._completed = value
        if value:
            mark_run_complete(self.conn, self.run_id)

    def mark_complete(self) -> None:
        """Mark dataset as complete and thus read only and notify the
        subscribers"""
        self.completed = True
        for sub in self.subscribers.values():
            sub.done_callback()

    def add_result(self, results: Dict[str, VALUE]) -> int:
        """
        Add a logically single result to existing parameters

        Args:
            - results: dictionary with name of a parameter as the key and the
               value to associate as the value.

        Returns:

            - index in the DataSet that the result was stored at

        If a parameter exist in the dataset and it's not in the results
        dictionary Null values are inserted.
        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to add results to a completed DataSet.
        """
        # TODO: Make this check less fugly
        for param in results.keys():
            if self.paramspecs[param].depends_on != '':
                deps = self.paramspecs[param].depends_on.split(', ')
                for dep in deps:
                    if dep not in results.keys():
                        raise ValueError(f'Can not add result for {param}, '
                                         f'since this depends on {dep}, '
                                         'which is not being added.')

        if self.completed:
            raise CompletedError
        index = insert_values(self.conn, self.table_name,
                              list(results.keys()),
                              list(results.values())
                              )
        return index

    def add_results(self, results: List[Dict[str, VALUE]]) -> int:
        """
        Adds a sequence of results to the DataSet.

        Args:
            - list of name, value dictionaries  where each
              dictionary provides the values for the parameters in
              that result. If some parameters are missing the corresponding
              values are assumed to be None

        Returns:
            - the index in the DataSet that the **first** result was stored at

        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to add results to a completed DataSet.
        """
        expected_keys = frozenset.union(*[frozenset(d) for d in results])
        values = [[d.get(k, None) for k in expected_keys] for d in results]

        len_before_add = length(self.conn, self.table_name)
        insert_many_values(self.conn, self.table_name, list(expected_keys),
                           values)
        return len_before_add

    def modify_result(self, index: int, results: Dict[str, VALUES]) -> None:
        """ Modify a logically single result of existing parameters

        Args:
            - index: zero-based index of the result to be modified.
            - results: dictionary of updates with name of a parameter as the
               key and the value to associate as the value.


        It is an error to modify a result at an index less than zero or
        beyond the end of the DataSet.
        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to modify a result in a completed DataSet.
        """
        if self.completed:
            raise CompletedError
        for param in results.keys():
            if param not in self.paramspecs.keys():
                raise ValueError(f'No such parameter: {param}.')
        with atomic(self.conn) as self.conn:
            modify_values(self.conn, self.table_name, index,
                          list(results.keys()),
                          list(results.values())
                          )

    def modify_results(self, start_index: int,
                       updates: List[Dict[str, VALUES]]):
        """ Modify a sequence of results in the DataSet.

        Args:
            - index: zero-based index of the result to be modified.
            - results: sequence of dictionares of updates with name of a
                parameter as the key and the value to associate as the value.


        It is an error to modify a result at an index less than zero or
        beyond the end of the DataSet.
        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to modify a result in a completed DataSet.
        """
        if self.completed:
            raise CompletedError

        keys = [list(val.keys()) for val in updates]
        flattened_keys = [item for sublist in keys for item in sublist]

        mod_params = set(flattened_keys)
        old_params = set(self.paramspecs.keys())
        if not mod_params.issubset(old_params):
            raise ValueError('Can not modify values for parameter(s) '
                             f'{mod_params.difference(old_params)}, '
                             'no such parameter(s) in the dataset.')

        values = [list(val.values()) for val in updates]
        flattened_values = [item for sublist in values for item in sublist]

        with atomic(self.conn) as self.conn:
            modify_many_values(self.conn,
                               self.table_name,
                               start_index,
                               flattened_keys,
                               flattened_values)

    def add_parameter_values(self, spec: ParamSpec, values: VALUES):
        """
        Add a parameter to the DataSet and associates result values with the
        new parameter.

        Adds a parameter to the DataSet and associates result values with the
        new parameter.
        If the DataSet is not empty, then the count of provided
        values must equal the current count of results in the DataSet, or an
        error will result.

        It is an error to add parameters to a completed DataSet.
        # TODO: fix type checking
        """
        # first check that the len of values (if dataset is not empty)
        # is the right size i.e. the same as the dataset
        if len(self) > 0:
            if len(values) != len(self):
                raise ValueError("Need to have {} values but got {}.".format(
                    len(self),
                    len(values)
                ))
        with atomic(self.conn) as self.conn:
            add_parameter(self.conn, self.table_name, spec)
            # now add values!
            results = [{spec.name: value} for value in values]
            self.add_results(results)

    def get_data(self,
                 *params: Union[str, ParamSpec, _BaseParameter],
                 start: Optional[int] = None,
                 end: Optional[int] = None) -> List[List[Any]]:
        """ Returns the values stored in the DataSet for the specified parameters.
        The values are returned as a list of lists, SQL rows by SQL columns,
        e.g. datapoints by parameters. The data type of each element is based on
        the datatype provided when the DataSet was created. The parameter list may
        contain a mix of string parameter names, QCoDeS Parameter objects, and
        ParamSpec objects. As long as they have a `name` field. If provided,
        the start and end parameters select a range of results by result count
        (index).
        If the range is empty -- that is, if the end is less than or
        equal to the start, or if start is after the current end of the
        DataSet – then a list of empty arrays is returned.

        For a more type independent and easier to work with view of the data
        you may want to consider using
        :py:meth:`qcodes.dataset.data_export.get_data_by_id`


        Args:
            - *params: string parameter names, QCoDeS Parameter objects, and
               ParamSpec objects
            - start:
            - end:

        Returns:
            - list of lists SQL rows of data by SQL columns. Each SQL row
              is a datapoint and each SQL column is a parameter.  Each element
              will be of the datatypes stored in the database
              (numeric, array or string)
        """
        valid_param_names = []
        for maybeParam in params:
            if isinstance(maybeParam, str):
                valid_param_names.append(maybeParam)
                continue
            else:
                try:
                    maybeParam = maybeParam.name
                except Exception as e:
                    raise ValueError(
                        "This parameter does not have  a name") from e
            valid_param_names.append(maybeParam)
        data = get_data(self.conn, self.table_name, valid_param_names,
                        start, end)
        return data

    def get_values(self, param_name: str) -> List[List[Any]]:
        """
        Get the values (i.e. not NULLs) of the specified parameter
        """
        if param_name not in self.parameters:
            raise ValueError('Unknown parameter, not in this DataSet')

        values = get_values(self.conn, self.table_name, param_name)

        return values

    def get_setpoints(self, param_name: str) -> Dict[str, List[List[Any]]]:
        """
        Get the setpoints for the specified parameter

        Args:
            param_name: The name of the parameter for which to get the
                setpoints
        """

        if param_name not in self.parameters:
            raise ValueError('Unknown parameter, not in this DataSet')

        if self.paramspecs[param_name].depends_on == '':
            raise ValueError(f'Parameter {param_name} has no setpoints.')

        setpoints = get_setpoints(self.conn, self.table_name, param_name)

        return setpoints

    # NEED to pass Any for some reason
    def subscribe(self,
                  callback: Callable[[Any, int, Optional[Any]], None],
                  min_wait: int = 0,
                  min_count: int = 1,
                  state: Optional[Any] = None,
                  callback_kwargs: Optional[Dict[str, Any]] = None
                  ) -> str:
        subscriber_id = uuid.uuid4().hex
        subscriber = _Subscriber(self, subscriber_id, callback, state, min_wait, min_count,
                                 callback_kwargs)
        self.subscribers[subscriber_id] = subscriber
        subscriber.start()
        return subscriber_id

    def unsubscribe(self, uuid: str) -> None:
        """
        Remove subscriber with the provided uuid
        """
        with atomic(self.conn) as self.conn:
            self._remove_trigger(uuid)
            sub = self.subscribers[uuid]
            sub.schedule_stop()
            sub.join()
            del self.subscribers[uuid]

    def _remove_trigger(self, name):
        transaction(self.conn, f"DROP TRIGGER IF EXISTS {name};")

    def unsubscribe_all(self):
        """
        Remove all subscribers
        """
        sql = "select * from sqlite_master where type = 'trigger';"
        triggers = atomic_transaction(self.conn, sql).fetchall()
        with atomic(self.conn) as self.conn:
            for trigger in triggers:
                self._remove_trigger(trigger['name'])
            for sub in self.subscribers.values():
                sub.schedule_stop()
                sub.join()
            self.subscribers.clear()

    def get_metadata(self, tag):
        return get_metadata(self.conn, tag, self.table_name)

    def __len__(self) -> int:
        return length(self.conn, self.table_name)

    def __repr__(self) -> str:
        out = []
        heading = f"{self.name} #{self.run_id}@{self.path_to_db}"
        out.append(heading)
        out.append("-" * len(heading))
        ps = self.get_parameters()
        if len(ps) > 0:
            for p in ps:
                out.append(f"{p.name} - {p.type}")

        return "\n".join(out)


# public api
def load_by_id(run_id)->DataSet:
    """ Load dataset by id

    Args:
        run_id: id of the dataset

    Returns:
        the datasets

    """
    d = DataSet(get_DB_location(), run_id=run_id)
    return d


def load_by_counter(counter, exp_id):
    """
    Load a dataset given its counter in one experiment
    Args:
        counter: Counter of the dataset
        exp_id:  Experiment the dataset belongs to

    Returns:
        the dataset
    """
    conn = connect(get_DB_location())
    sql = """
    SELECT run_id
    FROM
      runs
    WHERE
      result_counter= ? AND
      exp_id = ?
    """
    c = transaction(conn, sql, counter, exp_id)
    run_id = one(c, 'run_id')
    conn.close()
    d = DataSet(get_DB_location(), run_id=run_id)

    return d


def new_data_set(name, exp_id: Optional[int] = None,
                 specs: SPECS = None, values=None,
                 metadata=None, conn=None) -> DataSet:
    """ Create a new dataset.
    If exp_id is not specified the last experiment will be loaded by default.

    Args:
        name: the name of the new dataset
        exp_id:  the id of the experiments this dataset belongs to
            defaults to the last experiment
        specs: list of parameters to create this data_set with
        values: the values to associate with the parameters
        metadata:  the values to associate with the dataset
    """
    path_to_db = get_DB_location()
    if conn is None:
        tempcon = True
        conn = connect(get_DB_location())
    else:
        tempcon = False

    if exp_id is None:
        if len(get_experiments(conn)) > 0:
            exp_id = get_last_experiment(conn)
        else:
            raise ValueError("No experiments found."
                             "You can start a new one with:"
                             " new_experiment(name, sample_name)")
    # This is admittedly a bit weird. We create a dataset, link it to some
    # run in the DB and then (using _new) change what it's linked to
    if tempcon:
        conn.close()
        conn = None
    d = DataSet(path_to_db, run_id=None, conn=conn)
    d._new(name, exp_id, specs, values, metadata)

    return d


# helpers
def hash_from_parts(*parts: str) -> str:
    """
    Args:
        *parts:  parts to use to create hash
    Returns:
        hash created with the given parts
    """
    warnings.warn("hash_from_parts has been deprecated and will be removed. "
                  "Use stdlib uuid4 instead",
                  stacklevel=2)
    combined = "".join(parts)
    return hashlib.sha1(combined.encode("utf-8")).hexdigest()
