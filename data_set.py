#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 unga <giulioungaretti@me.com>
#
# Distributed under terms of the MIT license.

import atexit
import hashlib
import logging
import numpy as np
import time
from threading import Thread
from functools import partial
from itertools import count
from typing import List, Any, Dict, Union,  Callable, Optional


from qcodes.utils.metadata import Metadatable
from qcodes.instrument.parameter import _BaseParameter


def hash_from_parts(*parts: str) -> str:
    """
    Args:
        *parts:  parts to use to create hash

    Returns:
        hash created with the given parts

    """
    combined = "".join(parts)
    return hashlib.sha1(combined.encode("utf-8")).hexdigest()


# TODO: does it make sense to have the type here ?
# the question is mostly how to specifiy (dtypes from numpy wont'work)
# and as such there are just two types:
# scalar and not scalar
# TODO: units, unit, and all of that is not in the spec
class ParamSpec(Metadatable):
    def __init__(self, name: str, metadata=None) -> None:
        self.name = name
        self.id: str = hash_from_parts(name)
        # actual data
        self._data: List[Any] = []
        # id of associated setpoints if any
        self._setpoints = None

        super().__init__(metadata)

    @property
    def setpoints(self):
        return self._setpoints

    @setpoints.setter
    def setpoints(self, id):
        self._setpoints = id

    # TODO: data here or not? this is slightly confusing in the spec
    # as we call parameter the entity that hold the "row-like" result
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def add(self, value):
        """ add value to paramSpec

        Args:
            value:  any data (scalar, or really anything)
        """
        self._data.append(value)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "{}:{}".format(self.name, len(self.data))


def param_spec(parameter: _BaseParameter) -> ParamSpec:
    """ Generates a ParamSpec from a qcodes parameter

    Args:
        - parameter: the qcodes parameter to make a spec

    """
    return ParamSpec(parameter.name, parameter.metadata)


# SPECS is a list of ParamSpec
SPECS = List[ParamSpec]


class CompletedError(RuntimeError):
    pass


class DataSet(Metadatable):
    def __init__(self, name, specs: SPECS=None, values=None,
                 metadata=None) -> None:
        self.name: str = name
        self.id: str = hash_from_parts(name)

        self.subscribers: Dict[str, Subscriber] = {}

        self.completed: bool = False

        self.parameters: Dict[str, ParamSpec] = {}
        # make sure we clean up on exit
        atexit.register(self.unsubscribe_all)

        # counter to keep track of the lenght of the dataSet
        self.c = count(0)
        # init here so we have the metadata dictionary
        super().__init__(metadata)

        # add empty parameter entry
        self.metadata['parameters'] = {}
        if specs and values:
            self._pouplate_parameters(specs, values)
        elif specs:
            self.add_parameters(specs)

    def _pouplate_parameters(self, specs: SPECS,
                             values: List[list]) -> None:
        if len(specs) != len(values):
            raise ValueError("Expected same number of specs and values")
        for spec, value in zip(specs, values):
            spec.data = value
            self.add_parameter(spec)

    def add_parameter(self, spec: ParamSpec):
        """
        Mutates internal paraemter dictioanry
        """
        if self.completed:
            raise CompletedError
        else:
            # NOTE: (giulioungaretti)
            # don't think this will be nice json
            # but that's the spec people wanted
            self.parameters[spec.id] = spec
            self.metadata["parameters"].update({
                spec.name: {
                    'metadata': spec.metadata
                }
            })

    def get_parameter(self, name: str) -> ParamSpec:
        return self.parameters[hash_from_parts(name)]

    def add_parameters(self, specs: SPECS):
        for spec in specs:
            self.add_parameter(spec)

    def add_metadata(self, tag: str, metadata: Any):
        """
        Adds metadata to the DataSet. The metadata is stored under the
        provided tag.

        Args:
            tag: represents the key in the metadata dictionary
            metadata: actual metadata

        """
        self.metadata[tag] = metadata

    def mark_complete(self) -> None:
        """Mark dataset as complete and thus read only and notify the
        subscribers"""
        self.completed = True
        for sub in self.subscribers.values():
            sub.done_callback()

    def add_result(self, name: str, value: Any) -> int:
        """
        Add a logically single result to an existing parameter

        Args:
            - name: name of existing parameter
            - value: value to associate

        Returns:
            - index in the DataSet that the result was stored at

        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to add results to a completed DataSet.
        """
        if self.completed:
            raise CompletedError
        else:
            result = self.parameters[hash_from_parts(name)]
            _len = len(result)
            result.add(value)
            return _len

    def add_results(self, results: Dict[str, Any]) -> Dict[str, int]:
        """
        Add a logically single result to existing parameters

        Args:
            - results: dictionary with name of a parameter as the key and the
              value to associate as the value.

        Returns:

            - dictionary with name of a parameter as the key and the
              the index in the DataSet that the result were stored at

        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to add results to a completed DataSet.
        """
        items = results.items()
        length = [(name, self.add_result(name, value))
                  for name, value in items]
        return dict(length)

    # TODO: name feels horrible, but see notes on the spec file.
    def add_many(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Adds a sequence of results to the DataSet.

        Args:
            - list of name, value dictionaries  where each
              dictionary provides the values for all of the parameters in
              that result.

        Returns:
            - dictionary with name of a parameter as the key and the
              the index in the DataSet that the **first** result was stored at

        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to add results to a completed DataSet.
        """
        indices = map(self.add_results, results)
        return list(indices)[0]

    def modify_result(self, index: int, name: str, value: Any) -> None:
        """ Modify a logically single result of  an existing parameter

        Args:
            - index: zero-based index of the result to be modified.
            - name: name of paramSpec to modify
            - value: new value to assign,
                     set to None to remvoe form dataset


        It is an error to modify a result at an index less than zero or
        beyond the end of the DataSet.
        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to modify a result in a completed DataSet.
        """
        if self.completed:
            raise CompletedError

        param = self.get_parameter(name)

        if value is None:
            self.parameters.pop(hash_from_parts(name))
        else:
            param.data[index] = value

    def modify_results(self, index: int, updates: Dict[str, Any]):
        """ Modify a logically single result of existing parameters

        Args:
            - index: zero-based index of the result to be modified.
            - name: name of paramSpec to modify
            - value: new value to assign,
                        set to None to remvoe form dataset


        It is an error to modify a result at an index less than zero or
        beyond the end of the DataSet.
        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.
        It is an error to modify a result in a completed DataSet.
        """
        items = updates.items()
        _partial = partial(self.modify_result, index)
        for name, value in items:
            _partial(name, value)

    def modify_many(self, start_index: int,
                    updates: List[Dict[str, Any]]) -> None:
        """ Modifies a sequence of results in the DataSet.
        """
        # TODO: this is not a fast implematation 🦄
        if start_index + len(updates) > len(self):
            msg = "Modification excedes the boundary of this dataset"
            raise RuntimeError(msg)
        for i, update in enumerate(updates):
            self.modify_results(start_index + i, update)

    def add_parameter_values(self, spec: ParamSpec, values: Any):
        """
        Add a parameter to the DataSet and associates result values with the
        new parameter.
        """
        spec.add(values)
        self.add_parameter(spec)

    def get_param_setpoints(self, name):
        """
        Get all setpoints (nested) of this param
        """
        param = self.get_parameter(name)
        setpoints = []
        while param.setpoints:
            setpoint = self.parameters[param.setpoints]
            setpoints.append(setpoint)
            param = self.get_parameter(setpoint.name)
        return setpoints

    def _guess_shape(self, name):
        setpoints = self.get_param_setpoints(name)
        shape = list([len(setpoint) for setpoint in setpoints[::-1]])
        return shape

    def to_array(self, name, shape=None):
        if shape is None:
            shape = self._guess_shape(name)
        values = np.array(self.get_data(name))
        return values.reshape(shape)

    def get_data(self,
                 *params: Union[str, ParamSpec, _BaseParameter],
                 start: int=0,
                 end: int=-1)-> List[List[Any]]:
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
        data = []
        for name in valid_param_names:
            if end == -1:
                data.append(self.get_parameter(name).data[start:])
            else:
                data.append(self.get_parameter(name).data[start:end])
        return data

    def get_parameters(self)->List[ParamSpec]:
        return list(self.parameters.values())

    def get_metadata(self, tag):
        return self.metadata[tag]

    #  NEED to pass Any for some reason
    def subscribe(self, callback: Callable[[Any, int, Optional[Any]], None],
                  min_wait: int = 0, min_count: int=1,
                  state: Optional[Any]=None) -> str:
        sub_id = hash_from_parts(str(time.time()))
        sub = Subscriber(self, sub_id, callback, state, min_wait, min_count)
        sub.start()
        self.subscribers[sub_id] = sub
        return sub_id

    def unsubscribe(self, uuid: str)-> None:
        sub = self.subscribers[uuid]
        sub.schedule_stop()
        sub.join()
        del self.subscribers[uuid]

    def unsubscribe_all(self):
        for sub in self.subscribers.values():
            sub.schedule_stop()
            sub.join()
        self.subscribers.clear()

    def snapshot_base(self, update=False):
        return self.metadata

    def __len__(self):
        vals = self.parameters.values()
        return max(map(len, vals))

    def __repr__(self):
        out = []
        out.append(self.name)
        out.append("-" * len(self.name))
        for _, param in self.parameters.items():
            out.append(param.__repr__())
        return "\n".join(out)


# TODO: this is an example MINIMAL subscriber
# the specs names dataset obejct, so only threading 
# can be used, uneless one wants to end up again in 
# the pickle-hell-situation
class Subscriber(Thread):

    def __init__(self, data: DataSet, sub_id: str,
                 callback: Callable[[DataSet, int, Optional[Any]], None],
                 state: Optional[Any]=None, min_wait: int=100,
                 min_count: int=1)->None:
        self.min_wait = min_wait
        self.min_count = min_count
        self.sub_id = sub_id
        self._send_queue: int = 0
        self.data = data
        self.state = state
        self.callback = callback
        self._stop_signal: bool = False
        super().__init__()
        self.log = logging.getLogger(f"Subscriber {self.sub_id}")

    def run(self)->None:
        self.log.debug("Starting subscriber")
        self._loop()

    def _loop(self)->None:
        while True:
            if self._stop_signal:
                self._clean_up()
                break
            self._send_queue += len(self.data)
            if self._send_queue > self.min_count:
                self.callback(self.data, len(self.data), self.state)
                self._send_queue = 0
            # if nothing happens we let the word go foward
            time.sleep(self.min_wait/1000)
            if self.data.completed:
                break

    def done_callback(self)->None:
        self.log.debug("Done callback")
        self.callback(self.data, len(self.data), self.state)

    def schedule_stop(self):
        self.log.debug("Scheduling stop")
        if not self._stop_signal:
            self._stop_signal = True

    def _clean_up(self)->None:
        # TODO: just a temp implemation
        self.log.debug("Stopped subscriber")