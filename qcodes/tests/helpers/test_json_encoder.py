from collections import OrderedDict, UserDict
import json

import numpy as np
import pytest
from qcodes.utils.helpers import NumpyJSONEncoder


def test_python_types():
    e = NumpyJSONEncoder()

    # test basic python types
    od = OrderedDict()
    od['a'] = 0
    od['b'] = 1
    testinput = [None, True, False, 10, float(10.), 'hello',
                 od]
    testoutput = ['null', 'true', 'false', '10', '10.0', '"hello"',
                  '{"a": 0, "b": 1}']

    for d, r in zip(testinput, testoutput):
        v = e.encode(d)
        if isinstance(d, dict):
            assert v == r
        else:
            assert v == r


def test_complex_types():
    e = NumpyJSONEncoder()
    assert e.encode(complex(1, 2)) == \
           '{"__dtype__": "complex", "re": 1.0, "im": 2.0}'
    assert e.encode(np.complex(1, 2)) == \
           '{"__dtype__": "complex", "re": 1.0, "im": 2.0}'
    assert e.encode(np.complex64(complex(1, 2))) == \
           '{"__dtype__": "complex", "re": 1.0, "im": 2.0}'


def test_numpy_int_types():
    e = NumpyJSONEncoder()

    numpy_ints = (np.int, np.int_, np.int8, np.int16, np.int32,
                  np.int64, np.intc, np.intp,
                  np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
                  np.uintc, np.uintp)

    for int_type in numpy_ints:
        assert e.encode(int_type(3)) == '3'


def test_numpy_float_types():
    e = NumpyJSONEncoder()

    numpy_floats = (np.float, np.float_, np.float16, np.float32,
                    np.float64)

    for float_type in numpy_floats:
        assert e.encode(float_type(2.5)) == '2.5'


def test_numpy_bool_type():
    e = NumpyJSONEncoder()

    assert e.encode(np.bool_(True)) == 'true'
    assert e.encode(np.int8(5) == 5) == 'true'
    assert e.encode(np.array([8, 5]) == 5) == '[false, true]'


def test_numpy_array():
    e = NumpyJSONEncoder()

    assert e.encode(np.array([1, 0, 0])) == \
           '[1, 0, 0]'

    assert e.encode(np.arange(1.0, 3.0, 1.0)) == \
           '[1.0, 2.0]'

    assert e.encode(np.meshgrid((1, 2), (3, 4))) == \
           '[[[1, 2], [1, 2]], [[3, 3], [4, 4]]]'


def test_non_serializable():
    """
    Test that non-serializable objects are serialzed to their
    string representation
    """
    e = NumpyJSONEncoder()

    class Dummy:
        def __str__(self):
            return 'i am a dummy with \\ slashes /'

    assert e.encode(Dummy()) == \
           '"i am a dummy with \\\\ slashes /"'


def test_object_with_serialization_method():
    """
    Test that objects with `_JSONEncoder` method are serialized via
    calling that method
    """
    e = NumpyJSONEncoder()

    class Dummy:
        def __init__(self):
            self.confession = 'a_dict_addict'

        def __str__(self):
            return 'me as a string'

        def _JSONEncoder(self):
            return {'i_am_actually': self.confession}

    assert e.encode(Dummy()) == \
           '{"i_am_actually": "a_dict_addict"}'


class SomeUserDict(UserDict):
    pass

EXAMPLEMETADATA = {
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
    'verisimilitude': 1j,
    'myuserdict': SomeUserDict({'a': 1})
}


def test_standard_encoder_fails_examplemetadata():
    with pytest.raises(TypeError):
        json.dumps(
            EXAMPLEMETADATA,
            sort_keys=True,
            indent=4,
            ensure_ascii=False)


def test_numpy_encoder_examplemetadata():
    data = json.dumps(EXAMPLEMETADATA, sort_keys=True, indent=4,
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
        'verisimilitude': {'__dtype__': 'complex', 're': 0, 'im': 1},
        'myuserdict': {'a': 1}
    }
    assert metadata == data_dict
