from unittest import TestCase
import numpy as np
import json

from qcodes.utils.helpers import NumpyJSONEncoder


class TestNumpyJson(TestCase):

    def setUp(self):
        self.metadata = {
            'name': 'Rapunzel',
            'age': np.int64(12),
            'height': np.float64(112.234),
            'scores': np.linspace(0, 42, num=3),
            # include some regular values to ensure they work right
            # with our encoder
            'weight': 19,
            'length': 45.23,
            'points': [12, 24, 48],
            'RapunzelNumber': np.float64(4.89) + np.float64(0.11) * 1j,
            'verisimilitude': 1j
        }

    def test_numpy_fail(self):
        metadata = self.metadata
        with self.assertRaises(TypeError):
            json.dumps(metadata, sort_keys=True, indent=4, ensure_ascii=False)

    def test_numpy_good(self):
        metadata = self.metadata
        data = json.dumps(metadata, sort_keys=True, indent=4,
                          ensure_ascii=False, cls=NumpyJSONEncoder)
        data_dict = json.loads(data)

        metadata = {
            'name': 'Rapunzel',
            'age': 12,
            'height': 112.234,
            'scores': [0, 21, 42],
            'weight': 19,
            'length': 45.23,
            'points': [12, 24, 48],
            'RapunzelNumber': {'__dtype__': 'complex', 're': 4.89, 'im': 0.11},
            'verisimilitude': {'__dtype__': 'complex', 're': 0, 'im': 1}
        }

        self.assertEqual(metadata, data_dict)
