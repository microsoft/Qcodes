import math
import re
from unittest import TestCase

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import given
from qcodes.utils.types import (complex_types, numpy_concrete_floats,
                                numpy_concrete_ints, numpy_non_concrete_floats,
                                numpy_non_concrete_ints)
from qcodes.utils.validators import (Arrays, Callable,
                                     ComplexNumbers, Dict, Enum, Ints, Lists,
                                     Multiples, MultiType,
                                     PermissiveMultiples,
                                     Strings)


class AClass:

    def method_a(self):
        raise RuntimeError('function should not get called')


def a_func():
    pass


class TestEnum(TestCase):
    enums = [
        [True, False],
        [1, 2, 3],
        [True, None, 1, 2.3, 'Hi!', b'free', (1, 2), float("inf")]
    ]

    # enum items must be immutable - tuple OK, list bad.
    not_enums = [[], [[1, 2], [3, 4]]]

    def test_good(self):
        for enum in self.enums:
            e = Enum(*enum)

            for v in enum:
                e.validate(v)

            for v in [22, 'bad data', [44, 55]]:
                with pytest.raises((ValueError, TypeError)):
                    e.validate(v)

            assert repr(e) == '<Enum: {}>'.format(repr(set(enum)))

            # Enum is never numeric, even if its members are all numbers
            # because the use of is_numeric is for sweeping
            assert not e.is_numeric

    def test_bad(self):
        for enum in self.not_enums:
            with pytest.raises(TypeError):
                Enum(*enum)

    def test_valid_values(self):
        for enum in self.enums:
            e = Enum(*enum)
            for val in e._valid_values:
                e.validate(val)


class TestMultiples(TestCase):
    divisors = [3, 7, 11, 13]
    not_divisors = [0, -1, -5, -1e15, 0.1, -0.1, 1.0, 3.5,
                    -2.3e6, 5.5e15, 1.34e-10, -2.5e-5,
                    math.pi, math.e, '', None, float("nan"), float("inf"),
                    -float("inf"), '1', [], {}, [1, 2], {1: 1}, b'good',
                    AClass, AClass(), a_func]
    multiples = [0, 1, 10, -1, 100, 1000000, int(-1e15), int(1e15),
                 # warning: True==1 and False==0 - we *could* prohibit these, using
                 # isinstance(v, bool)
                 True, False,
                 # numpy scalars
                 np.int64(2)]
    not_multiples = [0.1, -0.1, 1.0, 3.5, -2.3e6, 5.5e15, 1.34e-10, -2.5e-5,
                     math.pi, math.e, '', None, float("nan"), float("inf"),
                     -float("inf"), '1', [], {}, [1, 2], {1: 1}, b'good',
                     AClass, AClass(), a_func]

    def test_divisors(self):
        for d in self.divisors:
            n = Multiples(divisor=d)
            for v in [d * e for e in self.multiples]:
                n.validate(v)

            for v in self.multiples:
                if v == 0:
                    continue
                with pytest.raises(ValueError):
                    n.validate(v)

            for v in self.not_multiples:
                with pytest.raises(TypeError):
                    n.validate(v)

        for d in self.not_divisors:
            with pytest.raises(TypeError):
                n = Multiples(divisor=d)

        n = Multiples(divisor=3, min_value=1, max_value=10)
        assert repr(n) == '<Ints 1<=v<=10, Multiples of 3>'

    def test_valid_values(self):

        for d in self.divisors:
            n = Multiples(divisor=d)
            for num in n.valid_values:
                n.validate(num)


class TestPermissiveMultiples(TestCase):
    divisors = [40e-9, -1, 0.2225, 1/3, np.pi/2]

    multiples = [[800e-9, -40e-9, 0, 1],
                 [3, -4, 0, -1, 1],
                 [1.5575, -167.9875, 0],
                 [2/3, 3, 1, 0, -5/3, 1e4],
                 [np.pi, 5*np.pi/2, 0, -np.pi/2]]

    not_multiples = [[801e-9, 4.002e-5],
                     [1.5, 0.9999999],
                     [0.2226],
                     [0.6667, 28/9],
                     [3*np.pi/4]]

    def test_passing(self):
        for divind, div in enumerate(self.divisors):
            val = PermissiveMultiples(div)
            for mult in self.multiples[divind]:
                val.validate(mult)

    def test_not_passing(self):
        for divind, div in enumerate(self.divisors):
            val = PermissiveMultiples(div)
            for mult in self.not_multiples[divind]:
                with pytest.raises(ValueError):
                    val.validate(mult)

    # finally, a quick test that the precision is indeed setable
    def test_precision(self):
        pm_lax = PermissiveMultiples(35e-9, precision=3e-9)
        pm_lax.validate(72e-9)
        pm_strict = PermissiveMultiples(35e-9, precision=1e-10)
        with pytest.raises(ValueError):
            pm_strict.validate(70.2e-9)

    def test_valid_values(self):
        for div in self.divisors:
            pm = PermissiveMultiples(div)
            for val in pm.valid_values:
                pm.validate(val)


class TestMultiType(TestCase):

    def test_good(self):
        m = MultiType(Strings(2, 4), Ints(10, 1000))

        for v in [10, 11, 123, 1000, 'aa', 'mop', 'FRED']:
            m.validate(v)

        for v in [9, 1001, 'Q', 'Qcode', None, 100.0, b'nice', [], {},
                  a_func, AClass, AClass(), True, False]:
            with pytest.raises(ValueError):
                m.validate(v)

        assert repr(m) == '<MultiType: Strings 2<=len<=4, Ints 10<=v<=1000>'

        assert m.is_numeric

        assert not MultiType(Strings(), Enum(1, 2)).is_numeric


    def test_bad(self):
        for args in [[], [1], [Strings(), True]]:
            with pytest.raises(TypeError):
                MultiType(*args)

    def test_valid_values(self):
        ms = [MultiType(Strings(2, 4), Ints(10, 32)),
              MultiType(Ints(), Lists(Ints())),
              MultiType(Ints(), MultiType(Lists(Ints()), Ints()))]
        for m in ms:
            for val in m.valid_values:
                m.validate(val)


class TestArrays(TestCase):

    def test_type(self):
        m = Arrays(min_value=0.0, max_value=3.2, shape=(2, 2))
        for v in ['somestring', 4, 2, [[2, 0], [1, 2]]]:
            with pytest.raises(TypeError):
                m.validate(v)

    def test_complex_min_max_raises(self):
        """
        Min max is not implemented for complex types
        """
        with pytest.raises(TypeError, match=r"min_value must be a real number\."
                                            r" It is \(1\+1j\) of type "
                                            r"<class 'complex'>"):
            Arrays(min_value=1+1j)
        with pytest.raises(TypeError, match=r"max_value must be a real number. "
                                            r"It is \(1\+1j\) of type "
                                            r"<class 'complex'>"):
             Arrays(max_value=1+1j)
        with pytest.raises(TypeError, match=r'Setting min_value or max_value is not '
                                            r'supported for complex validators'):
            Arrays(max_value=1, valid_types=(np.complexfloating,))

    def test_complex(self):
        a = Arrays(valid_types=(np.complexfloating, ))
        for dtype in complex_types:
            a.validate(np.arange(10, dtype=dtype))

    def test_complex_subtypes(self):
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

    def test_min_max_real_ints_raises(self):
        with pytest.raises(TypeError, match="min_value must be an instance "
                                            "of valid_types."):
            Arrays(valid_types=(np.integer,), min_value=1.0)
        with pytest.raises(TypeError, match="max_value must be an instance "
                                            "of valid_types."):
            Arrays(valid_types=(np.integer,), max_value=6.0)

    def test_min_max_ints_real_raises(self):
        with pytest.raises(TypeError, match="min_value must be an instance "
                                            "of valid_types."):
            Arrays(valid_types=(np.floating,), min_value=1)
        with pytest.raises(TypeError, match="max_value must be an instance "
                                            "of valid_types."):
            Arrays(valid_types=(np.floating,), max_value=6)

    def test_real_subtypes(self):
        """
        Test that validating a concrete real type into an array that
        only support other concrete types raises as expected
        """
        types = list(numpy_concrete_ints + numpy_concrete_floats)
        randint = np.random.randint(0, len(types))
        mytype = types.pop(randint)

        a = Arrays(valid_types=(mytype,))
        a.validate(np.arange(10, dtype=mytype))
        a = Arrays(valid_types=types)
        with pytest.raises(TypeError, match=r'is not any of'):
            a.validate(np.arange(10, dtype=mytype))

    def test_complex_default_raises(self):
        """Complex types should not validate by default"""
        a = Arrays()
        for dtype in complex_types:
            with pytest.raises(TypeError, match=r"is not any of \(<class "
                                                r"'numpy.integer'>, <class "
                                                r"'numpy.floating'>\) "
                                                r"it is complex"):
                a.validate(np.arange(10, dtype=dtype))

    def test_text_type_raises(self):
        """Text types are not supported """
        with pytest.raises(TypeError, match="Arrays validator only supports "
                                            "numeric types: <class "
                                            "'numpy.str_'> is not supported."):
            Arrays(valid_types=(np.dtype('<U5').type,))

    def test_text_array_raises(self):
        """Test that an array of text raises"""
        a = Arrays()
        with pytest.raises(TypeError,
                           match=r"type of \['A' 'BC' 'CDF'\] is not any of "
                                 r"\(<class 'numpy.integer'>, <class "
                                 r"'numpy.floating'>\) it is <U3;"):
            a.validate(np.array(['A', 'BC', 'CDF']))

    def test_default_types(self):
        """Arrays constructed with all concrete and abstract real number
        types should validate by default"""
        a = Arrays()

        integer_types = (int,) + numpy_non_concrete_ints + numpy_concrete_ints
        for mytype in integer_types:
            a.validate(np.arange(10, dtype=mytype))

        float_types = (float,) + numpy_non_concrete_floats \
                      + numpy_concrete_floats
        for mytype in float_types:
            a.validate(np.arange(10, dtype=mytype))

    def test_min_max(self):
        m = Arrays(min_value=-5, max_value=50, shape=(2, 2))
        v = np.array([[2, 0], [1, 2]])
        m.validate(v)
        v = 100*v
        with pytest.raises(ValueError):
            m.validate(v)
        v = -1*v
        with pytest.raises(ValueError):
            m.validate(v)

        m = Arrays(min_value=-5, shape=(2, 2))
        v = np.array([[2, 0], [1, 2]])
        m.validate(v*100)

    def test_max_smaller_min_raises(self):
        with pytest.raises(TypeError, match='max_value must be '
                                            'bigger than min_value'):
            Arrays(min_value=10, max_value=-10)

    def test_shape(self):
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

    def test_shape_defered(self):
        m = Arrays(min_value=-5, max_value=50, shape=(lambda: 2, lambda: 2))
        v1 = np.array([[2, 5], [3, 7]])
        m.validate(v1)
        v2 = np.array([[2, 0], [1, 2], [2, 3]])
        with pytest.raises(ValueError):
            m.validate(v2)

    def test_valid_values_with_shape(self):
        val = Arrays(min_value=-5, max_value=50, shape=(2, 2))
        for vval in val.valid_values:
            val.validate(vval)

    def test_valid_values(self):
        val = Arrays(min_value=-5, max_value=50)
        for vval in val.valid_values:
            val.validate(vval)

    def test_shape_non_sequence_raises(self):
        with pytest.raises(ValueError):
            m = Arrays(shape=5)
        with pytest.raises(ValueError):
            m = Arrays(shape=lambda: 10)

    def test_repr(self):
            a = Arrays()
            assert str(a) == '<Arrays, shape: None>'
            b = Arrays(min_value=1, max_value=2)
            assert str(b) == '<Arrays 1<=v<=2, shape: None>'
            c = Arrays(shape=(2, 2))
            assert str(c) == '<Arrays, shape: (2, 2)>'
            c = Arrays(shape=(lambda: 2, 2))
            assert re.match(r"<Arrays, shape: \(<function TestArrays."
                            r"test_repr.<locals>.<lambda> "
                            r"at 0x[a-fA-f0-9]*>, 2\)>", str(c))


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
