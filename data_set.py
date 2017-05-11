#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 unga <giulioungaretti@me.com>
#
# Distributed under terms of the MIT license.

import hashlib
import numpy as np
from typing import List, Dict

from qcodes.utils.metadata import Metadatable


def hash(*parts):
    combined = "".join(parts)
    return hashlib.sha1(combined.encode("utf-8")).hexdigest()

# Maybe this shoudl be a named tuple?
class ParamSpec(Metadatable):

    def __init__(self, name, metadata=None):
        self.name = name
        self.id = hash(name)
        # actualy data
        self._data = []
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
        self._data.append(value)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "{}:{}".format(self.name, len(self.data))


# types are cool
SPECS = List[ParamSpec]

class CompletedError(RuntimeError):
    pass

class DataSet(Metadatable):


    def __init__(self, name,
                 specs: SPECS= None,
                 values=None,
                 metadata=None) -> None:


        self.name = name
        self.id = hashlib.sha1(name.encode("utf-8")).hexdigest()

        self.completed: bool = False

        self.parameters: Dict[str, ParamSpec] = {}
        if specs and values:
            self.parameters = self._pouplate_parameters(specs, values)
        elif specs:
            self.add_parameters(specs)


        super().__init__(metadata)

    @staticmethod
    def _pouplate_parameters(specs: SPECS, values: List[list]):
        parameters = {}
        if len(specs) != len(values):
            raise ValueError("Expected same unmber of specs and values")
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
        return self.parameters[hash(name)]

    def add_parameters(self, specs: SPECS):
        for spec in specs:
            self.add_parameter(spec)

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
                self.parameters[hash(name)].add(value)

    def add_results(self, results):
        """
        TODO not sure this makes sesne
        """
        self.add_result(**results)

    def add_parameter_values(self, spec, values):
        """
        Add a parameter to the DataSet and associates result values with the
        new parameter.
        """
        spec.add(values)
        self.add_parameter(spec)
    
    def get_data(self, name):
        return self.get_parameter(name).data

    def get_param_setpoints(self, name) :
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
        shape = list([len(setpoint)for setpoint in setpoints[::-1]])
        return shape

    def to_array(self, name):
        # assume the data has the right shape
        shape = self.guess_shape(name)
        values = np.array(self.get_data(name))
        return values.reshape(shape)

    def __repr__(self):
        out = []
        out.append(self.name)
        out.append("-"*len(self.name))
        for _, param in self.parameters.items():
            out.append(param.__repr__())
        return "\n".join(out)

