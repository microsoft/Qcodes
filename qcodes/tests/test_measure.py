from unittest import TestCase
from datetime import datetime

from qcodes.instrument.parameter import ManualParameter
from qcodes.measure import Measure

from .instrument_mocks import MultiGetter


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
        self.assertFalse(meta['background'])
        self.assertFalse(meta['use_data_manager'])
        self.assertFalse(meta['use_threads'])

        ts_start = datetime.strptime(meta['ts_start'], '%Y-%m-%d %H:%M:%S')
        ts_end = datetime.strptime(meta['ts_end'], '%Y-%m-%d %H:%M:%S')
        self.assertGreaterEqual(ts_end, ts_start)

    def test_simple_array(self):
        data = Measure(MultiGetter(arr=(1.2, 3.4))).run_temp()

        self.assertEqual(data.index0.tolist(), [0, 1])
        self.assertEqual(data.arr.tolist(), [1.2, 3.4])
        self.assertEqual(len(data.arrays), 2, data.arrays)

    def test_array_and_scalar(self):
        self.p1.set(42)
        data = Measure(MultiGetter(arr=(5, 6)), self.p1).run_temp()

        self.assertEqual(data.single_set.tolist(), [0])
        self.assertEqual(data.P1.tolist(), [42])
        self.assertEqual(data.index0.tolist(), [0, 1])
        self.assertEqual(data.arr.tolist(), [5, 6])
        self.assertEqual(len(data.arrays), 4, data.arrays)
