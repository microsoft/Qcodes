from unittest import TestCase
from datetime import datetime

from qcodes.data.location import FormatLocation
from qcodes.instrument.parameter import Parameter
from qcodes.measure import Measure

from ..instrument_mocks import MultiGetter, MultiSetPointParam

import numpy as np
from numpy.testing import assert_array_equal

class TestMeasure(TestCase):
    def setUp(self):
        self.p1 = Parameter('P1', initial_value=1, get_cmd=None, set_cmd=None)

    def test_simple_scalar(self):
        data = Measure(self.p1).run_temp()

        self.assertEqual(data.single_set.tolist(), [0])
        self.assertEqual(data.P1.tolist(), [1])
        self.assertEqual(len(data.arrays), 2, data.arrays)

        self.assertNotIn('loop', data.metadata)

        meta = data.metadata['measurement']
        self.assertEqual(meta['__class__'], 'qcodes.measure.Measure')
        self.assertEqual(len(meta['actions']), 1)
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
        loc_fmt = 'data/{date}/#{counter}_{name}_{date}_{time}'
        rcd = {'name': 'test_metadata'}
        param_name_1 = "multi_setpoint_param_this"
        param_name_2 = "multi_setpoint_param_that"
        setpoint_name = "multi_setpoint_param_this_setpoint_set"
        loc_provider = FormatLocation(fmt=loc_fmt, record=rcd)
        c = Measure(self.p1).run(location=loc_provider)
        self.assertEqual(c.metadata['arrays'][param_name_1]['unit'], 'this unit')
        self.assertEqual(c.metadata['arrays'][param_name_1]['name'], param_name_1)
        self.assertEqual(c.metadata['arrays'][param_name_1]['label'], 'this label')
        self.assertEqual(c.metadata['arrays'][param_name_1]['is_setpoint'], False)
        self.assertEqual(c.metadata['arrays'][param_name_1]['shape'], (5,))
        assert_array_equal(getattr(c, param_name_1).ndarray, np.zeros(5))

        self.assertEqual(c.metadata['arrays'][param_name_2]['unit'], 'that unit')
        self.assertEqual(c.metadata['arrays'][param_name_2]['name'], param_name_2)
        self.assertEqual(c.metadata['arrays'][param_name_2]['label'], 'that label')
        self.assertEqual(c.metadata['arrays'][param_name_2]['is_setpoint'], False)
        self.assertEqual(c.metadata['arrays'][param_name_2]['shape'], (5,))
        assert_array_equal(getattr(c, param_name_2).ndarray, np.ones(5))

        self.assertEqual(c.metadata['arrays'][setpoint_name]['unit'],
                         'this setpointunit')
        self.assertEqual(c.metadata['arrays'][setpoint_name]['name'],
                         "multi_setpoint_param_this_setpoint")
        self.assertEqual(c.metadata['arrays'][setpoint_name]['label'],
                         'this setpoint')
        self.assertEqual(c.metadata['arrays'][setpoint_name]
                         ['is_setpoint'], True)
        self.assertEqual(c.metadata['arrays'][setpoint_name]['shape'],
                         (5,))
        assert_array_equal(getattr(c, setpoint_name).ndarray, np.linspace(5, 9, 5))
