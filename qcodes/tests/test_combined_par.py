from collections import OrderedDict

import unittest
from unittest.mock import patch
from unittest.mock import call

from hypothesis import given
import hypothesis.strategies as hst

import numpy as np

from qcodes.instrument.parameter import combine
from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import full_class


class DumyPar(Metadatable):

    """Docstring for DumyPar. """

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.full_name = name

    def __str__(self):
        return self.full_name

    def set(self, value):
        value = value * 2
        return value


class TestMultiPar(unittest.TestCase):

    def setUp(self):
        parameters = [DumyPar(name) for name in ["X", "Y", "Z"]]
        self.parameters = parameters
        self.input_dimensionality = len(parameters)

    def testCombine(self):
        multipar = combine(*self.parameters, name="combined")
        self.assertEqual(multipar.dimensionality,
                         self.input_dimensionality)

    def testSweepBadSetpoints(self):
        with self.assertRaises(ValueError):
            combine(*self.parameters, name="fail").sweep(np.array([[1, 2]]))

    def testSweep(self):
        setpoints = np.array([[1, 1, 1], [1, 1, 1]])

        sweep_values = combine(*self.parameters,
                               name="combined").sweep(setpoints)

        res = []
        for i in sweep_values:
            value = sweep_values.set(i)
            res.append([i, value])
        expected = [
                [0, [1, 1, 1]],
                [1, [1, 1, 1]]
                ]
        self.assertEqual(res, expected)

    def testSet(self):
        setpoints = np.array([[1, 1, 1], [1, 1, 1]])

        sweep_values = combine(*self.parameters,
                               name="combined").sweep(setpoints)

        with patch.object(sweep_values, 'set') as mock_method:
            for i in sweep_values:
                    sweep_values.set(i)

        mock_method.assert_has_calls([
                    call(0), call(1)
                ]
            )


    @given(npoints=hst.integers(1, 100),
           x_start_stop=hst.lists(hst.integers(), min_size=2, max_size=2).map(sorted),
           y_start_stop=hst.lists(hst.integers(), min_size=2, max_size=2).map(sorted),
           z_start_stop=hst.lists(hst.integers(), min_size=2, max_size=2).map(sorted))
    def testAggregator(self, npoints, x_start_stop, y_start_stop, z_start_stop):

        x_set = np.linspace(x_start_stop[0], x_start_stop[1], npoints).reshape(npoints, 1)
        y_set = np.linspace(y_start_stop[0], y_start_stop[1], npoints).reshape(npoints, 1)
        z_set = np.linspace(z_start_stop[0], z_start_stop[1], npoints).reshape(npoints, 1)
        setpoints = np.hstack((x_set, y_set, z_set))
        expected_results = [linear(*set) for set in setpoints]
        sweep_values = combine(*self.parameters,
                               name="combined",
                               aggregator=linear).sweep(setpoints)

        results = []
        for i, value in enumerate(sweep_values):
                res = sweep_values.set(value)
                results.append(sweep_values._aggregate(*res))

        self.assertEqual(results, expected_results)

    def testMeta(self):
        name = "combined"
        label = "Linear Combination"
        unit = "a.u"
        aggregator = linear
        sweep_values = combine(*self.parameters,
                               name=name,
                               label=label,
                               unit=unit,
                               aggregator=aggregator
                               )
        snap = sweep_values.snapshot()
        out = OrderedDict()
        out['__class__'] = full_class(sweep_values)
        out["unit"] = unit
        out["label"] = label
        out["full_name"] = name
        out["aggregator"] = repr(linear)
        for param in sweep_values.parameters:
            out[param.full_name] = {}
        self.assertEqual(out, snap)

    def testMutable(self):
        setpoints = np.array([[1, 1, 1], [1, 1, 1]])

        sweep_values = combine(*self.parameters,
                               name="combined")
        a = sweep_values.sweep(setpoints)
        setpoints = np.array([[2, 1, 1], [1, 1, 1]])
        b = sweep_values.sweep(setpoints)
        self.assertNotEqual(a, b)

    def testArrays(self):
        x_vals = np.linspace(1, 1, 2)
        y_vals = np.linspace(1, 1, 2)
        z_vals = np.linspace(1, 1, 2)
        sweep_values = combine(*self.parameters,
                               name="combined").sweep(x_vals, y_vals, z_vals)
        res = []
        for i in sweep_values:
            value = sweep_values.set(i)
            res.append([i, value])

        expected = [
                [0, [1, 1, 1]],
                [1, [1, 1, 1]]
                ]
        self.assertEqual(res, expected)

    def testWrongLen(self):
        x_vals = np.linspace(1, 1, 2)
        y_vals = np.linspace(1, 1, 2)
        z_vals = np.linspace(1, 1, 3)
        with self.assertRaises(ValueError):
            combine(*self.parameters,
                    name="combined").sweep(x_vals, y_vals, z_vals)


    def testInvalidName(self):
        x_vals = np.linspace(1, 1, 2)
        y_vals = np.linspace(1, 1, 2)
        z_vals = np.linspace(1, 1, 2)
        with self.assertRaises(ValueError):
            combine(*self.parameters,
                    name="combined with spaces").sweep(x_vals, y_vals, z_vals)

    def testLen(self):
        x_vals = np.linspace(1, 1, 2)
        y_vals = np.linspace(1, 1, 2)
        z_vals = np.linspace(1, 0, 2)
        sweep_values = combine(*self.parameters,
                               name="combined").sweep(x_vals, y_vals, z_vals)
        self.assertEqual(len(x_vals), len(sweep_values.setpoints))


def linear(x, y, z):
    return x+y+z
