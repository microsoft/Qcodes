from unittest import TestCase

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import given
from qcodes.utils.validators import (Callable,
                                     ComplexNumbers, Dict, Ints, Lists)


class AClass:

    def method_a(self):
        raise RuntimeError('function should not get called')


def a_func():
    pass


class TestLists(TestCase):

    def test_type(self):
        l = Lists()
        v1 = ['a', 'b', 5]
        l.validate(v1)

        v2 = 234
        with pytest.raises(TypeError):
            l.validate(v2)

    def test_elt_vals(self):
        l = Lists(Ints(max_value=10))
        v1 = [0, 1, 5]
        l.validate(v1)

        v2 = [0, 1, 11]
        with pytest.raises(ValueError):
            l.validate(v2)

    def test_valid_values(self):
        val = Lists(Ints(max_value=10))
        for vval in val.valid_values:
            val.validate(vval)


class TestCallable(TestCase):

    def test_callable(self):
        c = Callable()

        def test_func():
            return True
        c.validate(test_func)
        test_int = 5
        with pytest.raises(TypeError):
            c.validate(test_int)

    def test_valid_values(self):
        val = Callable()
        for vval in val.valid_values:
            val.validate(vval)


class TestDict(TestCase):

    def test_dict(self):
        d = Dict()
        test_dict = {}
        d.validate(test_dict)
        test_int = 5
        with pytest.raises(TypeError):
            d.validate(test_int)

    def test_valid_values(self):
        val = Dict()
        for vval in val.valid_values:
            val.validate(vval)


@given(complex_val=hst.complex_numbers())
def test_complex(complex_val):

    n = ComplexNumbers()
    assert str(n) == '<Complex Number>'
    n.validate(complex_val)
    n.validate(np.complex(complex_val))
    n.validate(np.complex64(complex_val))
    n.validate(np.complex128(complex_val))


@given(val=hst.one_of(hst.floats(), hst.integers(), hst.characters()))
def test_complex_raises(val):

    n = ComplexNumbers()

    with pytest.raises(TypeError, match=r"is not complex;"):
        n.validate(val)
