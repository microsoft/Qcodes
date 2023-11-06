import re

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import complex_number_dtypes

from qcodes.utils.types import (
    concrete_complex_types,
    numpy_concrete_ints,
    numpy_floats,
    numpy_non_concrete_ints_instantiable,
)
from qcodes.validators import Arrays


def test_type() -> None:
    m = Arrays(min_value=0.0, max_value=3.2, shape=(2, 2))
    for v in ['somestring', 4, 2, [[2, 0], [1, 2]]]:
        with pytest.raises(TypeError):
            m.validate(v)  # type: ignore[arg-type]


def test_complex_min_max_raises() -> None:
    """
    Min max is not implemented for complex types
    """
    with pytest.raises(TypeError, match=r"min_value must be a real number\."
                                        r" It is \(1\+1j\) of type "
                                        r"<class 'complex'>"):
        Arrays(min_value=1 + 1j)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"max_value must be a real number. "
                                        r"It is \(1\+1j\) of type "
                                        r"<class 'complex'>"):
        Arrays(max_value=1 + 1j)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r'Setting min_value or max_value is '
                                        r'not supported for complex '
                                        r'validators'):
        Arrays(max_value=1, valid_types=(np.complexfloating,))


@given(dtype=complex_number_dtypes())
def test_complex(dtype: np.dtype) -> None:
    a = Arrays(valid_types=(np.complexfloating,))
    a.validate(np.arange(10, dtype=dtype))


def test_complex_subtypes() -> None:
    """Test that specifying a specific complex subtype works as expected"""
    a = Arrays(valid_types=(np.complex64,))

    a.validate(np.arange(10, dtype=np.complex64))
    with pytest.raises(TypeError, match=r"is not any of "
                                        r"\(<class 'numpy.complex64'>,\)"
                                        r" it is complex128"):
        a.validate(np.arange(10, dtype=np.complex128))
    a = Arrays(valid_types=(np.complex128,))

    a.validate(np.arange(10, dtype=np.complex128))
    with pytest.raises(TypeError, match=r"is not any of "
                                        r"\(<class 'numpy.complex128'>,\)"
                                        r" it is complex64"):
        a.validate(np.arange(10, dtype=np.complex64))


def test_min_max_real_ints_raises() -> None:
    with pytest.raises(TypeError, match="min_value must be an instance "
                                        "of valid_types."):
        Arrays(valid_types=(np.integer,), min_value=1.0)
    with pytest.raises(TypeError, match="max_value must be an instance "
                                        "of valid_types."):
        Arrays(valid_types=(np.integer,), max_value=6.0)


def test_min_max_ints_real_raises() -> None:
    with pytest.raises(TypeError, match="min_value must be an instance "
                                        "of valid_types."):
        Arrays(valid_types=(np.floating,), min_value=1)
    with pytest.raises(TypeError, match="max_value must be an instance "
                                        "of valid_types."):
        Arrays(valid_types=(np.floating,), max_value=6)


def test_real_subtypes() -> None:
    """
    Test that validating a concrete real type into an array that
    only support other concrete types raises as expected
    """
    types = list(set(numpy_concrete_ints + numpy_floats))
    randint = np.random.randint(0, len(types))
    mytype = types.pop(randint)

    a = Arrays(valid_types=(mytype,))
    a.validate(np.arange(10, dtype=mytype))
    a = Arrays(valid_types=types)
    with pytest.raises(TypeError, match=r'is not any of'):
        a.validate(np.arange(10, dtype=mytype))


def test_complex_default_raises() -> None:
    """Complex types should not validate by default"""
    a = Arrays()
    for dtype in concrete_complex_types:
        with pytest.raises(TypeError, match=r"is not any of \(<class "
                                            r"'numpy.integer'>, <class "
                                            r"'numpy.floating'>\) "
                                            r"it is complex"):
            a.validate(np.arange(10, dtype=dtype))


def test_text_type_raises() -> None:
    """Text types are not supported """
    with pytest.raises(TypeError, match="Arrays validator only supports "
                                        "numeric types: <class "
                                        "'numpy.str_'> is not supported."):
        Arrays(valid_types=(np.dtype('<U5').type,))


def test_text_array_raises() -> None:
    """Test that an array of text raises"""
    a = Arrays()
    with pytest.raises(TypeError,
                       match=r"type of \['A' 'BC' 'CDF'\] is not any of "
                             r"\(<class 'numpy.integer'>, <class "
                             r"'numpy.floating'>\) it is <U3;"):
        a.validate(np.array(['A', 'BC', 'CDF']))


def test_default_types() -> None:
    """Arrays constructed with all concrete and abstract real number
    types should validate by default"""
    a = Arrays()

    integer_types = (
            (int,)
            + numpy_non_concrete_ints_instantiable
            + numpy_concrete_ints
    )
    for mytype in integer_types:
        a.validate(np.arange(10, dtype=mytype))

    float_types = (float,) + numpy_floats
    for mytype in float_types:
        a.validate(np.arange(10, dtype=mytype))


def test_min_max() -> None:
    m = Arrays(min_value=-5, max_value=50, shape=(2, 2))
    v = np.array([[2, 0], [1, 2]])
    m.validate(v)
    v = 100 * v
    with pytest.raises(ValueError):
        m.validate(v)
    v = -1 * v
    with pytest.raises(ValueError):
        m.validate(v)

    m = Arrays(min_value=-5, shape=(2, 2))
    v = np.array([[2, 0], [1, 2]])
    m.validate(v * 100)


def test_max_smaller_min_raises() -> None:
    with pytest.raises(TypeError, match='max_value must be '
                                        'bigger than min_value'):
        Arrays(min_value=10, max_value=-10)


def test_shape() -> None:
    m = Arrays(min_value=-5, max_value=50, shape=(2, 2))

    v1 = np.array([[2, 0], [1, 2]])
    v2 = np.array([[2, 0], [1, 2], [2, 3]])

    # v1 is the correct shape but v2 is not
    m.validate(v1)
    with pytest.raises(ValueError):
        m.validate(v2)
    # both should pass if no shape specified
    m = Arrays(min_value=-5, max_value=50)
    m.validate(v1)
    m.validate(v2)


def test_shape_defered() -> None:
    m = Arrays(min_value=-5, max_value=50, shape=(lambda: 2, lambda: 2))
    v1 = np.array([[2, 5], [3, 7]])
    m.validate(v1)
    v2 = np.array([[2, 0], [1, 2], [2, 3]])
    with pytest.raises(ValueError):
        m.validate(v2)


def test_valid_values_with_shape() -> None:
    val = Arrays(min_value=-5, max_value=50, shape=(2, 2))
    for vval in val.valid_values:
        val.validate(vval)


def test_valid_values() -> None:
    val = Arrays(min_value=-5, max_value=50)
    for vval in val.valid_values:
        val.validate(vval)


def test_shape_non_sequence_raises() -> None:
    with pytest.raises(ValueError):
        _ = Arrays(shape=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _ = Arrays(shape=lambda: 10)  # type: ignore[arg-type]


def test_repr() -> None:
    a = Arrays()
    assert str(a) == '<Arrays, shape: None>'
    b = Arrays(min_value=1, max_value=2)
    assert str(b) == '<Arrays 1<=v<=2, shape: None>'
    c = Arrays(shape=(2, 2))
    assert str(c) == '<Arrays, shape: (2, 2)>'
    c = Arrays(shape=(lambda: 2, 2))
    assert re.match(
        r"<Arrays, shape: \(<function "
        r"test_repr.<locals>.<lambda> "
        r"at 0x[a-fA-F0-9]*>, 2\)>",
        str(c),
    )
