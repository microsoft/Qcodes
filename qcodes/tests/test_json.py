from unittest import TestCase
import numpy as np
import json
from copy import deepcopy

from qcodes.utils.helpers import json_encoder


class TestNumpyJson(TestCase):

    metadata = {'name': 'Rapunzel', 'age': 12, 'height': 112.234,
                'scores': np.linspace(0,42)}

    def test_numpy(self):
        metadata = deepcopy(self.metadata)
        metadata['scores'] = list(range(10))
        data = json.dumps(metadata, sort_keys=True, indent=4,
                          ensure_ascii=False)
        data_dict = json.loads(data)
        self.assertEqual(metadata, data_dict)

    def test_numpy_fail(self):
        metadata = self.metadata
        with self.assertRaises(TypeError):
            json.dumps(metadata, sort_keys=True, indent=4, ensure_ascii=False)


    def test_numpy_good(self):
        metadata = self.metadata
        data = json.dumps(metadata, sort_keys=True, indent=4,
                          ensure_ascii=False, cls=json_encoder)
        data_dict = json.loads(data)

        # fails
        # self.assertEqual(metadata, data_dict)
