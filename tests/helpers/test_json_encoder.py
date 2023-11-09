import json
from collections import OrderedDict, UserDict

import hypothesis.strategies as hst
import numpy as np
import pytest
import uncertainties  # type: ignore[import-untyped]
from hypothesis import given

from qcodes.utils import NumpyJSONEncoder
from qcodes.utils.types import numpy_complex, numpy_floats, numpy_ints


def test_python_types() -> None:
    e = NumpyJSONEncoder()

    # test basic python types
    od = OrderedDict()
    od["a"] = 0
    od["b"] = 1
    testinput = [None, True, False, 10, 10.0, "hello", od]
    testoutput = ["null", "true", "false", "10", "10.0", '"hello"', '{"a": 0, "b": 1}']

    for d, r in zip(testinput, testoutput):
        v = e.encode(d)
        if isinstance(d, dict):
            assert v == r
        else:
            assert v == r


def test_complex_types() -> None:
    e = NumpyJSONEncoder()
    assert e.encode(complex(1, 2)) == \
           '{"__dtype__": "complex", "re": 1.0, "im": 2.0}'
    for complex_type in numpy_complex:
        assert e.encode(complex_type(complex(1, 2))) == \
               '{"__dtype__": "complex", "re": 1.0, "im": 2.0}'


def test_UFloat_type() -> None:
    e = NumpyJSONEncoder()
    assert e.encode(uncertainties.ufloat(1.0, 2.0)) == \
           '{"__dtype__": "UFloat", "nominal_value": 1.0, "std_dev": 2.0}'


def test_numpy_int_types() -> None:
    e = NumpyJSONEncoder()

    for int_type in numpy_ints:
        assert e.encode(int_type(3)) == '3'


def test_numpy_float_types() -> None:
    e = NumpyJSONEncoder()

    for float_type in numpy_floats:
        assert e.encode(float_type(2.5)) == '2.5'


def test_numpy_bool_type() -> None:
    e = NumpyJSONEncoder()

    assert e.encode(np.bool_(True)) == 'true'
    assert e.encode(np.int8(5) == 5) == 'true'
    assert e.encode(np.array([8, 5]) == 5) == '[false, true]'


def test_numpy_array() -> None:
    e = NumpyJSONEncoder()

    assert e.encode(np.array([1, 0, 0])) == \
           '[1, 0, 0]'

    assert e.encode(np.arange(1.0, 3.0, 1.0)) == \
           '[1.0, 2.0]'

    assert e.encode(np.meshgrid((1, 2), (3, 4))) == \
           '[[[1, 2], [1, 2]], [[3, 3], [4, 4]]]'


def test_non_serializable() -> None:
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


def test_object_with_serialization_method() -> None:
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


def test_standard_encoder_fails_examplemetadata() -> None:
    with pytest.raises(TypeError):
        json.dumps(
            EXAMPLEMETADATA,
            sort_keys=True,
            indent=4,
            ensure_ascii=False)


def test_numpy_encoder_examplemetadata() -> None:
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


@given(bytes_to_encode=hst.binary(max_size=100))
def test_bytes(bytes_to_encode) -> None:
    e = NumpyJSONEncoder()
    v = e.encode(bytes_to_encode)

    default_encoder = json.JSONEncoder()
    # encoding bytes is the same as using the
    # default encode on the str value of the bytes
    assert v == default_encoder.encode(str(bytes_to_encode))


@given(bytes_to_encode=hst.binary(max_size=100))
def test_bytes_array(bytes_to_encode) -> None:
    e = NumpyJSONEncoder()
    value = bytearray(bytes_to_encode)
    v = e.encode(value)

    default_encoder = json.JSONEncoder()
    # encoding bytes is the same as using the
    # default encode on the str value of the bytes array
    assert v == default_encoder.encode(str(value))
