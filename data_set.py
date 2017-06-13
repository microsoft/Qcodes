#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 unga <giulioungaretti@me.com>
#
# Distributed under terms of the MIT license.

import hashlib
import numpy as np
from typing import List, Any

from qcodes.utils.metadata import Metadatable
from qcodes.instrument.parameter import _BaseParameter


def hash_from_parts(parts: str) -> str:
    """
    Args:
        *parts:  parts to use to create hash

    Returns:
        hash created with the given parts

    """
    combined = "".join(parts)
    return hashlib.sha1(combined.encode("utf-8")).hexdigest()


# TODO: does it make sense to have the type here.
# the question is mostly how to specifiy (dtypes from numpy wont'work)
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


def paramSpec(parameter: _BaseParameter) -> ParamSpec:
    return ParamSpec(parameter.name, parameter.metadata)

# types are cool
SPECS = List[ParamSpec]


class CompletedError(RuntimeError):
    pass


class DataSet(Metadatable):
    def __init__(self, name,
                 specs: SPECS = None,
                 values=None,
                 metadata=None) -> None:
        self.name: str = name
        self.id = hash_from_parts(name)

        self.completed = False

        self.parameters = {}
        if specs and values:
            self.parameters = self._pouplate_parameters(specs, values)
        elif specs:
            self.add_parameters(specs)

        super().__init__(metadata)

    @staticmethod
    def _pouplate_parameters(specs: SPECS, values: List[list]):
        parameters = {}
        if len(specs) != len(values):
            raise ValueError("Expected same number of specs and values")
        for spec, value in zip(specs, values):
            spec.data = value
            parameters[spec.id] = spec
        return parameters

    def add_parameter(self, spec: ParamSpec):
        if self.completed:
            raise CompletedError
        else:
            self.parameters[spec.id] = spec

    def get_parameter(self, name: str) -> ParamSpec:
        return self.parameters[hash_from_parts(name)]

    def add_parameters(self, specs: SPECS):
        for spec in specs:
            self.add_parameter(spec)

    def add_metadata(self, tag: str, metadata: object):
        """
        Adds metadata to the DataSet. The metadata is stored under the provided tag.

        Args:
            tag: represents the key in the metadata dictionary
            metadata: actual metadata
        """
        self.metadata[tag] = metadata

    def mark_complete(self):
        "Mark dataset as complete and thus read only"
        self.completed = True

    def add_result(self, **results):
        """
        Add a logically single result to an existing parameter
        """
        if self.completed:
            raise CompletedError
        else:
            for name, value in results.items():
                self.parameters[hash_from_parts(name)].add(value)

    def add_results(self, results):
        """
        TODO not sure this makes sesne
        """
        self.add_result(**results)

    def modify_result(self, index, name, value):
        if self.completed:
            raise CompletedError

        param = self.get_parameter(name)

        if value is None:
            self.parameters.pop(hash_from_parts(name))
        else:
            param.data[index] = value

    def add_parameter_values(self, spec, values):
        """
        Add a parameter to the DataSet and associates result values with the
        new parameter.
        """
        spec.add(values)
        self.add_parameter(spec)

    def get_data(self, name):
        return self.get_parameter(name).data

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

    def guess_shape(self, name):
        setpoints = self.get_param_setpoints(name)
        shape = list([len(setpoint) for setpoint in setpoints[::-1]])
        return shape

    def to_array(self, name):
        # assume the data has the right shape
        shape = self.guess_shape(name)
        values = np.array(self.get_data(name))
        return values.reshape(shape)

    def get_data(*params, start=0, end=-1):
        """
       Returns the values stored in the DataSet for the specified parameters.
       The values are returned as a list of parallel NumPy arrays, one array per parameter.
       The data type of each array is based on the data type provided when the DataSet was created.
       The parameter list may contain a mix of string parameter names, QCoDeS Parameter objects, and ParamSpec objects.
        As long as they have a `name` field.
       If provided, the start and end parameters select a range of results by result count (index).
        If the range is empty -- that is, if the end is less than or equal to the start, or if start is after the current end of the DataSet – then a list of empty arrays is returned.
        Args:
            *params:
            start:
            end:

        Returns:

        """
        valid_param_names = []
        for maybeparam in params:
            if isinstance(maybeParam, str):
                continue
            else:
                try:
                    maybeparam = maybeParam.name
                except Exception as e:
                    raise ValueError("This parameter does not have  a name") from e
            valid_param_names.append(maybeparam)
        return valid_param_names

    def __repr__(self):
        out = []
        out.append(self.name)
        out.append("-" * len(self.name))
        for _, param in self.parameters.items():
            out.append(param.__repr__())
        return "\n".join(out)
