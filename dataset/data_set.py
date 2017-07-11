#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 unga <giulioungaretti@me.com>
#
# Distributed under terms of the MIT license.
# import json
from typing import Any, Dict, List, Optional, Union, Sized, Callable
from threading import Thread
import time
import logging
import hashlib
from queue import Queue, Empty
from param_spec import ParamSpec
from qcodes.instrument.parameter import _BaseParameter
from sqlite_base import (atomic, atomicTransaction, transaction, add_parameter,
                         connect, create_run, get_parameters,
                         get_last_experiment, _select_one_where,
                         length, modify_values, add_meta_data, mark_run,
                         modify_many_values, insert_values, insert_many_values,
                         VALUES, get_data, get_metadata)


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


# TODO: this clearly should be configurable
DB = "/Users/unga/Desktop/experiment.db"


class Subscriber(Thread):
    def __init__(self, dataSet, id: str,
                 callback: Callable[[List[Any], int, Optional[Any]], None],
                 state: Optional[Any]=None, min_wait: int=100,
                 min_count: int=1)->None:
        self.sub_id = id
        # wether or not this is actually thread safe I am not sure :P
        self.dataSet = dataSet
        self.table_name = dataSet.table_name
        self.conn = dataSet.conn
        self.log = logging.getLogger(f"Subscriber {self.sub_id}")

        self.state = state
        self.min_wait = min_wait
        self.min_count = min_count
        self._send_queue: int = 0
        self.callback = callback
        self._stop_signal: bool = False

        parameters = dataSet.get_parameters()
        param_sql = ",".join([f"NEW.{p.name}" for p in parameters])
        self.callbackid = f"callback{self.sub_id}"

        self.conn.create_function(self.callbackid, -1, self.cache)
        sql = f"""
        CREATE TRIGGER sub{self.sub_id}
            AFTER INSERT ON '{self.table_name}'
        BEGIN
            SELECT {self.callbackid}({param_sql});
        END;"""
        atomicTransaction(self.conn, sql)
        self.data: Queue = Queue()
        self._data_set_len = len(dataSet)
        super().__init__()

    def cache(self, *args)->None:
        self.log.debug(f"{self.callbackid} called with args:{args}")
        self.data.put(args)
        self._data_set_len += 1
        self._send_queue += 1

    def run(self)->None:
        self.log.debug("Starting subscriber")
        self._loop()

    @staticmethod
    def _exhaust_queue(queue)->List:
        result_list = []
        while True:
            try:
                result_list.append(queue.get(block=False))
            except Empty:
                break
        return result_list

    def _send(self) -> List:
        result_list = self._exhaust_queue(self.data)
        self.callback(result_list, self._data_set_len, self.state)
        return result_list

    def _loop(self)->None:
        while True:
            if self._stop_signal:
                self._clean_up()
                break
            if self._send_queue > self.min_count:
                self._send()
                self._send_queue = 0

            # if nothing happens we let the word go foward
            time.sleep(self.min_wait/1000)
            if self.dataSet.completed:
                self._send()
                break

    def done_callback(self)->None:
        self.log.debug("Done callback")
        self._send()

    def schedule_stop(self):
        if not self._stop_signal:
            self.log.debug("Scheduling stop")
            self._stop_signal = True

    def _clean_up(self)->None:
        # TODO: just a temp implemation (remove?)
        self.log.debug("Stopped subscriber")


class DataSet(Sized):
    def __init__(self, path_to_db: str) -> None:
        # TODO: handle fail here by defaulting to
        # a standard db
        self.path_to_db = path_to_db
        self.conn = connect(self.path_to_db)
        self._debug = False

    def _new(self, name, exp_id, specs: SPECS=None, values=None,
             metadata=None) -> None:
        """
        Actually perform all the side effects needed for
        the creation of a new dataset.
        """
        counter, id, table_name = create_run(self.conn, exp_id, name,
                                             specs, values, metadata)

        # this is really the UUID (an ever increasing count in the db)
        self.id = id
        self.subscribers: Dict[str, Subscriber] = {}
        self._completed = False

    @property
    def name(self):
        return _select_one_where(self.conn, "runs",
                                 "name", "run_id", self.id)

    @property
    def table_name(self):
        return _select_one_where(self.conn, "runs",
                                 "result_table_name", "run_id", self.id)

    @property
    def counter(self):
        return _select_one_where(self.conn, "runs",
                                 "result_counter", "run_id", self.id)

    @property
    def exp_id(self):
        return _select_one_where(self.conn, "runs",
                                 "exp_id", "run_id", self.id)

    def toggle_debug(self):
        """
        Toggle debug mode, if debug mode is on
        all the queries made are echoed back.
        """
        self._debug = not self._debug
        self.conn.close()
        self.conn = connect(self.path_to_db, self._debug)

    def add_parameter(self, spec: ParamSpec):
        add_parameter(self.conn, self.table_name, spec)

    def get_parameters(self) -> SPECS:
        return get_parameters(self.conn, self.table_name)

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
        # add_meta_data(self.conn, self.id, {"metadata": json_meta_data})

        add_meta_data(self.conn, self.id, {tag: metadata})
        # adding meta-data does not commit
        self.conn.commit()

    @property
    def completed(self)->bool:
        return self._completed

    @completed.setter
    def completed(self, value):
        self._completed = value
        mark_run(self.conn, self.id, value)

    def mark_complete(self) -> None:
        """Mark dataset as complete and thus read only and notify the
        subscribers"""
        self.completed = True
        for sub in self.subscribers.values():
            sub.done_callback()

    def add_result(self, results: Dict[str, VALUES]) -> int:
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
        if self.completed:
            raise CompletedError
        index = insert_values(self.conn, self.table_name,
                              list(results.keys()),
                              list(results.values())
                              )
        self.conn.commit()
        return index

    def add_results(self, results: List[Dict[str, VALUES]]) -> int:
        """
        Adds a sequence of results to the DataSet.

        Args:
            - list of name, value dictionaries  where each
              dictionary provides the values for all of the parameters in
              that result.

        Returns:
            - the index in the DataSet that the **first** result was stored at

        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to add results to a completed DataSet.
        """
        keys = [list(val.keys()) for val in results]
        flattened_keys = [item for sublist in keys for item in sublist]
        values = [list(val.values()) for val in results]
        flattened_values = [item for sublist in values for item in sublist]
        len_before_add = length(self.conn, self.table_name)
        insert_many_values(self.conn, self.table_name, flattened_keys,
                           flattened_values)
        self.conn.commit()
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
        with atomic(self.conn):
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
        keys = [list(val.keys()) for val in updates]
        flattened_keys = [item for sublist in keys for item in sublist]
        values = [list(val.values()) for val in updates]
        flattened_values = [item for sublist in values for item in sublist]
        with atomic(self.conn):
                modify_many_values(self.conn,
                                   self.table_name,
                                   start_index,
                                   flattened_keys,
                                   flattened_values)

    def add_parameter_values(self, spec: ParamSpec, values: List[VALUES]):
        """
        Add a parameter to the DataSet and associates result values with the
        new parameter.

        Adds a parameter to the DataSet and associates result values with the
        new parameter.
        If the DataSet is not empty, then the count of provided
        values must equal the current count of results in the DataSet, or an
        error will result.

        It is an error to add parameters to a completed DataSet.
        # TODO: fix type cheking
        """
        # first check that the len of values (if dataset is not empty)
        # is the right size i.e. the same as the dataset
        if len(self) > 0:
            if len(values) != len(self):
                raise ValueError("Need to have {} values but got {}.".format(
                                  len(self),
                                  len(values)
                                  ))
        with atomic(self.conn):
            add_parameter(self.conn, self.table_name, spec)
            # now add values!
            results = [{spec.name: value} for value in values]
            self.add_results(results)

    def get_data(self,
                 *params: Union[str, ParamSpec, _BaseParameter],
                 start: Optional[int]= None,
                 end: Optional[int]= None)-> List[List[Any]]:
        """ Returns the values stored in the DataSet for the specified parameters.
        The values are returned as a list of parallel NumPy arrays, one array
        per parameter. The data type of each array is based on the data type
        provided when the DataSet was created. The parameter list may contain
        a mix of string parameter names, QCoDeS Parameter objects, and
        ParamSpec objects. As long as they have a `name` field. If provided,
        the start and end parameters select a range of results by result count
        (index).
        If the range is empty -- that is, if the end is less than or
        equal to the start, or if start is after the current end of the
        DataSet – then a list of empty arrays is returned.

        Args:
            - *params: string parameter names, QCoDeS Parameter objects, and
               ParamSpec objects
            - start:
            - end:

        Returns:
            - list of parallel NumPy arrays, one array per parameter
        per parameter
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

    # NEED to pass Any for some reason
    def subscribe(self, callback: Callable[[Any, int, Optional[Any]], None],
                  min_wait: int = 0, min_count: int=1,
                  state: Optional[Any]=None,
                  subscriber_class=Subscriber) -> str:
        sub_id = hash_from_parts(str(time.time()))
        sub = Subscriber(self, sub_id, callback, state, min_wait, min_count)
        self.subscribers[sub_id] = sub
        sub.start()
        return sub.sub_id

    def unsubscribe(self, uuid: str)-> None:
        """
        Remov subscriber with the provided uuid
        """
        with atomic(self.conn):
            self._remove_trigger(uuid)
            sub = self.subscribers[uuid]
            sub.schedule_stop()
            sub.join()
            del self.subscribers[uuid]

    def _remove_trigger(self, name):
        transaction(self.conn, f"DROP TRIGGER IF EXISTS name;")

    def unsubscribe_all(self):
        """
        Remove all subscribers
        """
        sql = "select * from sqlite_master where type = 'trigger';"
        triggers = atomicTransaction(self.conn, sql).fetchall()
        with atomic(self.conn):
            for trigger in triggers:
                self._remove_trigger(trigger['name'])
            for sub in self.subscribers.values():
                sub.schedule_stop()
                sub.join()
            self.subscribers.clear()

    def get_metadata(self, tag):
        return get_metadata(self.conn, tag, self.table_name)

    def __len__(self)->int:
        return length(self.conn, self.table_name)

    def __repr__(self)->str:
        out = []
        heading = f"{self.name} #{self.id}@{self.path_to_db}"
        out.append(heading)
        out.append("-" * len(heading))
        ps = self.get_parameters()
        if len(ps) > 0:
            for p in ps:
                out.append(f"{p.name} - {p.type}")

        return "\n".join(out)


# public api
def load_by_id(id):
    d = DataSet(DB)
    d.id = id
    return d


def load_by_counter(counter, exp_id):
    d = DataSet(DB)
    d.id = id
    return d


def new_data_set(name, exp_id: Optional[int]= None,
                 specs: SPECS=None, values=None,
                 metadata=None) -> DataSet:
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
    d = DataSet(DB)
    if exp_id is None:
        exp_id = get_last_experiment(d.conn)
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
    combined = "".join(parts)
    return hashlib.sha1(combined.encode("utf-8")).hexdigest()
