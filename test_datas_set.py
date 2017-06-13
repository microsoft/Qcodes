#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 unga <giulioungaretti@me.com>
#
# Distributed under terms of the MIT license.
import unittest

from data_set import ParamSpec, DataSet, CompletedError
from qcodes.tests.instrument_mocks import  MockParabola

NAME = "name"
HASH = "6ae999552a0d2dca14d62e2bc8b764d377b1dd6c"
SETPOINT_HASH = "7ae999552a0d2dca14d62e2bc8b764d377b1dd6c"


class TestParam(unittest.TestCase):

    def test_constructor(self):

        paramspec = ParamSpec(NAME)

        self.assertEqual(paramspec.name, "name")
        self.assertEqual(paramspec.metadata, {})
        self.assertEqual(paramspec.id, HASH)

    def test_setpoints(self):
        paramspec = ParamSpec(NAME)
        paramspec.setpoints = SETPOINT_HASH
        self.assertEqual(paramspec.setpoints, SETPOINT_HASH)

    def test_data(self):
        paramspec = ParamSpec(NAME)
        # add one value to the data list
        paramspec.add(1)
        self.assertEqual(len(paramspec.data), 1)

class TestDataSet(unittest.TestCase):

    def test_constructor(self):
        dataSet = DataSet(NAME)
        self.assertEqual(dataSet.name, "name")
        self.assertEqual(dataSet.metadata, {})
        self.assertEqual(dataSet.id, HASH)
        self.assertEqual(dataSet.parameters, {})
        self.assertEqual(dataSet.completed, False)

    def test_specs(self):
        paramspec = ParamSpec(NAME)
        dataSet = DataSet(NAME, [paramspec])
        parameters = {}
        parameters[HASH] = paramspec
        self.assertEqual(dataSet.parameters, parameters)


    def test_specs_values(self):
        paramspecs = [ParamSpec(NAME)]
        values = [[1, 2, 3]]
        dataSet = DataSet(NAME, paramspecs, values)
        expected ={}
        expected[HASH] = paramspecs[0]
        self.assertEqual(dataSet.parameters,expected)
        self.assertEqual(dataSet.parameters[HASH].data,[1,2,3])


    def test_add_spec(self):
        paramspec = ParamSpec(NAME)
        dataSet = DataSet(NAME)
        dataSet.add_parameter(paramspec)
        self.assertEqual(len(dataSet.parameters),1)
        dataSet.mark_complete()
        self.assertEqual(dataSet.completed,True)
        with self.assertRaises(CompletedError):
            dataSet.add_parameter(paramspec)

    def test_add_result(self):
        dataSet = DataSet(NAME)
        with self.assertRaises(KeyError):
            dataSet.add_result(name=0)
        paramspec = ParamSpec(NAME)
        dataSet.add_parameter(paramspec)
        for i in range(3):
            dataSet.add_result(name=i)
        self.assertEqual(dataSet.parameters[HASH].data,list(range(3)))

    def test_add_results(self):
        dataSet = DataSet(NAME)
        with self.assertRaises(KeyError):
            dataSet.add_result(name=0)
        paramspec = ParamSpec(NAME)
        dataSet.add_parameter(paramspec)
        for i in range(3):
            dataSet.add_results({"name":i})
        self.assertEqual(dataSet.parameters[HASH].data,list(range(3)))

    def test_set_points(self):
        x = ParamSpec("x")
        y = ParamSpec("y")
        y.setpoints = x.id
        z = ParamSpec("z")
        z.setpoints = y.id
        values = [[1, 2, 3],[3,4,6], [7,8,9]]
        dataSet = DataSet(NAME, [x,y,z], values)
        setpoints = dataSet.get_param_setpoints("z")
        self.assertEqual(setpoints,[y,x])

    def test_modify(self):
        x = ParamSpec("x")
        y = ParamSpec("y")
        y.setpoints = x.id
        z = ParamSpec("z")
        z.setpoints = y.id
        values = [[1, 2, 3],[3,4,6], [7,8,9]]
        dataSet = DataSet(NAME, [x,y,z], values)
        self.assertEqual(dataSet.get_parameter("x").data[0], 1)
        # set 0th point of result x to 0
        dataSet.modify_result(0,"x", 0)
        self.assertEqual(dataSet.get_parameter("x").data[0], 0)

    def test_modify_remove(self):
        x = ParamSpec("x")
        y = ParamSpec("y")
        y.setpoints = x.id
        z = ParamSpec("z")
        z.setpoints = y.id
        values = [[1, 2, 3],[3,4,6], [7,8,9]]
        dataSet = DataSet(NAME, [x,y,z], values)
        self.assertEqual(dataSet.get_parameter("x").data[0], 1)
        # remove x
        dataSet.modify_result(0,"x", None)
        with self.assertRaises(KeyError):
            dataSet.get_parameter("x")

    def test_get_data(self):
        qc_param = MockParabola("parabola")
        x = ParamSpec("x")
        y = ParamSpec("y")
        y.setpoints = x.id
        z = ParamSpec("z")
        z.setpoints = y.id
        values = [[1, 2, 3],[3,4,6], [7,8,9]]
        dataSet = DataSet(NAME, [x,y,z], values)
        dataSet.modify_result(0,"x", None)
        with self.assertRaises(KeyError):
            dataSet.get_parameter("x")

