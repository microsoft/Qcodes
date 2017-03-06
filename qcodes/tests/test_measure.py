from unittest import TestCase
from datetime import datetime

from qcodes.instrument.parameter import ManualParameter
from qcodes.measure import Measure

from .instrument_mocks import MultiGetter, MultiSetPointParam

import numpy as np
from numpy.testing import assert_array_equal

class TestMeasure(TestCase):
    def setUp(self):
        self.p1 = ManualParameter('P1', initial_value=1)

    def test_simple_scalar(self):
        data = Measure(self.p1).run_temp()

        self.assertEqual(data.single_set.tolist(), [0])
        self.assertEqual(data.P1.tolist(), [1])
        self.assertEqual(len(data.arrays), 2, data.arrays)

        self.assertNotIn('loop', data.metadata)

        meta = data.metadata['measurement']
        self.assertEqual(meta['__class__'], 'qcodes.measure.Measure')
        self.assertEqual(len(meta['actions']), 1)
        self.assertFalse(meta['use_data_manager'])
        self.assertFalse(meta['use_threads'])

        ts_start = datetime.strptime(meta['ts_start'], '%Y-%m-%d %H:%M:%S')
        ts_end = datetime.strptime(meta['ts_end'], '%Y-%m-%d %H:%M:%S')
        self.assertGreaterEqual(ts_end, ts_start)

    def test_simple_array(self):
        data = Measure(MultiGetter(arr=(1.2, 3.4))).run_temp()

        self.assertEqual(data.index0_set.tolist(), [0, 1])
        self.assertEqual(data.arr.tolist(), [1.2, 3.4])
        self.assertEqual(len(data.arrays), 2, data.arrays)

    def test_array_and_scalar(self):
        self.p1.set(42)
        data = Measure(MultiGetter(arr=(5, 6)), self.p1).run_temp()

        self.assertEqual(data.single_set.tolist(), [0])
        self.assertEqual(data.P1.tolist(), [42])
        self.assertEqual(data.index0_set.tolist(), [0, 1])
        self.assertEqual(data.arr.tolist(), [5, 6])
        self.assertEqual(len(data.arrays), 4, data.arrays)


class TestMeasureMulitParameter(TestCase):
    def setUp(self):
        self.p1 = MultiSetPointParam()


    def test_metadata(self):
        c = Measure(self.p1).run()
        self.assertEqual(c.metadata['arrays']['this']['unit'], 'this unit')
        self.assertEqual(c.metadata['arrays']['this']['name'], 'this')
        self.assertEqual(c.metadata['arrays']['this']['label'], 'this label')
        self.assertEqual(c.metadata['arrays']['this']['is_setpoint'], False)
        self.assertEqual(c.metadata['arrays']['this']['shape'], (5,))
        assert_array_equal(c.this.ndarray, np.zeros(5))

        self.assertEqual(c.metadata['arrays']['that']['unit'],'that unit')
        self.assertEqual(c.metadata['arrays']['that']['name'], 'that')
        self.assertEqual(c.metadata['arrays']['that']['label'], 'that label')
        self.assertEqual(c.metadata['arrays']['that']['is_setpoint'], False)
        self.assertEqual(c.metadata['arrays']['that']['shape'], (5,))
        assert_array_equal(c.that.ndarray, np.ones(5))

        self.assertEqual(c.metadata['arrays']['this_setpoint_set']['unit'], 'this setpointunit')
        self.assertEqual(c.metadata['arrays']['this_setpoint_set']['name'], 'this_setpoint')
        self.assertEqual(c.metadata['arrays']['this_setpoint_set']['label'], 'this setpoint')
        self.assertEqual(c.metadata['arrays']['this_setpoint_set']['is_setpoint'], True)
        self.assertEqual(c.metadata['arrays']['this_setpoint_set']['shape'], (5,))
        assert_array_equal(c.this_setpoint_set.ndarray, np.linspace(5, 9, 5))