import pytest

import numpy as np

from qcodes.instrument.parameter import Parameter
import qcodes.utils.validators as vals
from .conftest import BookkeepingValidator


def test_number_of_validations():
    p = Parameter('p', set_cmd=None, initial_value=0,
                  vals=BookkeepingValidator())
    # in the set wrapper the final value is validated
    # and then subsequently each step is validated.
    # in this case there is one step so the final value
    # is validated twice.
    assert p.vals.values_validated == [0, 0]

    p.step = 1
    p.set(10)
    assert p.vals.values_validated == [0, 0, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_number_of_validations_for_set_cache():
    p = Parameter('p', set_cmd=None,
                  vals=BookkeepingValidator())
    assert p.vals.values_validated == []

    p.cache.set(1)
    assert p.vals.values_validated == [1]

    p.cache.set(4)
    assert p.vals.values_validated == [1, 4]

    p.step = 1
    p.cache.set(10)
    assert p.vals.values_validated == [1, 4, 10]


def test_bad_validator():
    with pytest.raises(TypeError):
        Parameter('p', vals=[1, 2, 3])


def test_setting_int_with_float():
    parameter = Parameter(name='foobar', set_cmd=None, get_cmd=None,
                          set_parser=lambda x: int(round(x)),
                          vals=vals.PermissiveInts(0))

    a = 0
    b = 10
    values = np.linspace(a, b, b-a+1)
    for i in values:
        parameter(i)
        a = parameter()
        assert isinstance(a, int)


def test_setting_int_with_float_not_close():
    parameter = Parameter(name='foobar', set_cmd=None, get_cmd=None,
                          set_parser=lambda x: int(round(x)),
                          vals=vals.PermissiveInts(0))

    a = 0
    b = 10
    values = np.linspace(a, b, b-a+2)
    for i in values[1:-2]:
        with pytest.raises(TypeError):
            parameter(i)
