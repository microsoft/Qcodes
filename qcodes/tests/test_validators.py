from unittest import TestCase
import math
import numpy as np

from qcodes.utils.validators import (Validator, Anything, Bool, Strings,
                                     Numbers, Ints, PermissiveInts,
                                     Enum, MultiType, PermissiveMultiples,
                                     Arrays, Multiples, Lists, Callable, Dict)


class AClass:

    def method_a(self):
        raise RuntimeError('function should not get called')


def a_func():
    pass


class TestBaseClass(TestCase):

    def test_instantiate(self):
        # you cannot instantiate the base class
        with self.assertRaises(NotImplementedError):
            Validator()

    class BrokenValidator(Validator):

        def __init__(self):
            pass

    def test_broken(self):
        # nor can you call validate without overriding it in a subclass
        b = self.BrokenValidator()
        with self.assertRaises(NotImplementedError):
            b.validate(0)


class TestAnything(TestCase):

    def test_real_anything(self):
        a = Anything()
        for v in [None, 0, 1, 0.0, 1.2, '', 'hi!', [1, 2, 3], [],
                  {'a': 1, 'b': 2}, {}, set([1, 2, 3]), a, range(10),
                  True, False, float("nan"), float("inf"), b'good',
                  AClass, AClass(), a_func]:
            a.validate(v)

        self.assertEqual(repr(a), '<Anything>')

    def test_failed_anything(self):
        with self.assertRaises(TypeError):
            Anything(1)

        with self.assertRaises(TypeError):
            Anything(values=[1, 2, 3])


class TestBool(TestCase):
    bools = [True, False, np.bool8(True), np.bool8(False)]
    not_bools = [0, 1, 10, -1, 100, 1000000, int(-1e15), int(1e15),
                 0.1, -0.1, 1.0, 3.5, -2.3e6, 5.5e15, 1.34e-10, -2.5e-5,
                 math.pi, math.e, '', None, float("nan"), float("inf"),
                 -float("inf"), '1', [], {}, [1, 2], {1: 1}, b'good',
                 AClass, AClass(), a_func]

    def test_bool(self):
        b = Bool()

        for v in self.bools:
            b.validate(v)

        for v in self.not_bools:
            with self.assertRaises(TypeError):
                b.validate(v)

        self.assertEqual(repr(b), '<Boolean>')

    def test_valid_values(self):
        val = Bool()
        val.validate(val.valid_values[0])


class TestStrings(TestCase):
    long_string = '+'.join(str(i) for i in range(100000))
    danish = '\u00d8rsted F\u00e6lled'
    chinese = '\u590f\u65e5\u7545\u9500\u699c\u5927\u724c\u7f8e'

    strings = ['', '0', '10' '1.0e+10', 'a', 'Ja', 'Artichokes!',
               danish, chinese, long_string]

    not_strings = [0, 1, 1.0e+10, bytes('', 'utf8'),
                   bytes(danish, 'utf8'), bytes(chinese, 'utf8'),
                   [], [1, 2, 3], {}, {'a': 1, 'b': 2},
                   True, False, None, AClass, AClass(), a_func]

    def test_unlimited(self):
        s = Strings()

        for v in self.strings:
            s.validate(v)

        for v in self.not_strings:
            with self.assertRaises(TypeError):
                s.validate(v)

        self.assertEqual(repr(s), '<Strings>')

    def test_min(self):
        for min_len in [0, 1, 5, 10, 100]:
            s = Strings(min_length=min_len)
            for v in self.strings:
                if len(v) >= min_len:
                    s.validate(v)
                else:
                    with self.assertRaises(ValueError):
                        s.validate(v)

        for v in self.not_strings:
            with self.assertRaises(TypeError):
                s.validate(v)

        self.assertEqual(repr(s), '<Strings len>=100>')

    def test_max(self):
        for max_len in [1, 5, 10, 100]:
            s = Strings(max_length=max_len)
            for v in self.strings:
                if len(v) <= max_len:
                    s.validate(v)
                else:
                    with self.assertRaises(ValueError):
                        s.validate(v)

        for v in self.not_strings:
            with self.assertRaises(TypeError):
                s.validate(v)

        self.assertEqual(repr(s), '<Strings len<=100>')

    def test_range(self):
        s = Strings(1, 10)

        for v in self.strings:
            if 1 <= len(v) <= 10:
                s.validate(v)
            else:
                with self.assertRaises(ValueError):
                    s.validate(v)

        for v in self.not_strings:
            with self.assertRaises(TypeError):
                s.validate(v)

        self.assertEqual(repr(s), '<Strings 1<=len<=10>')

        # single-valued range
        self.assertEqual(repr(Strings(10, 10)), '<Strings len=10>')

    def test_failed_strings(self):
        with self.assertRaises(TypeError):
            Strings(1, 2, 3)

        with self.assertRaises(TypeError):
            Strings(10, 9)

        with self.assertRaises(TypeError):
            Strings(max_length=0)

        with self.assertRaises(TypeError):
            Strings(min_length=1e12)

        for length in [-1, 3.5, '2', None]:
            with self.assertRaises(TypeError):
                Strings(max_length=length)

            with self.assertRaises(TypeError):
                Strings(min_length=length)

    def test_valid_values(self):
        val = Strings()
        val.validate(val.valid_values[0])


class TestNumbers(TestCase):
    numbers = [0, 1, -1, 0.1, -0.1, 100, 1000000, 1.0, 3.5, -2.3e6, 5.5e15,
               1.34e-10, -2.5e-5, math.pi, math.e,
               # warning: True==1 and False==0
               True, False,
               # warning: +/- inf are allowed if max & min are not specified!
               -float("inf"), float("inf"),
               # numpy scalars
               np.int64(36), np.float32(-1.123)
               ]
    not_numbers = ['', None, '1', [], {}, [1, 2], {1: 1},
                   b'good', AClass, AClass(), a_func]

    def test_unlimited(self):
        n = Numbers()

        for v in self.numbers:
            n.validate(v)

        for v in self.not_numbers:
            with self.assertRaises(TypeError):
                n.validate(v)

        # special case - nan now raises a ValueError rather than a TypeError
        with self.assertRaises(ValueError):
            n.validate(float('nan'))

        n.validate(n.valid_values[0])

    def test_min(self):
        for min_val in [-1e20, -1, -0.1, 0, 0.1, 10]:
            n = Numbers(min_value=min_val)

            n.validate(n.valid_values[0])
            for v in self.numbers:
                if v >= min_val:
                    n.validate(v)
                else:
                    with self.assertRaises(ValueError):
                        n.validate(v)

        for v in self.not_numbers:
            with self.assertRaises(TypeError):
                n.validate(v)

        with self.assertRaises(ValueError):
            n.validate(float('nan'))

    def test_max(self):
        for max_val in [-1e20, -1, -0.1, 0, 0.1, 10]:
            n = Numbers(max_value=max_val)

            n.validate(n.valid_values[0])
            for v in self.numbers:
                if v <= max_val:
                    n.validate(v)
                else:
                    with self.assertRaises(ValueError):
                        n.validate(v)

        for v in self.not_numbers:
            with self.assertRaises(TypeError):
                n.validate(v)

        with self.assertRaises(ValueError):
            n.validate(float('nan'))

    def test_range(self):
        n = Numbers(0.1, 3.5)

        for v in self.numbers:
            if 0.1 <= v <= 3.5:
                n.validate(v)
            else:
                with self.assertRaises(ValueError):
                    n.validate(v)

        for v in self.not_numbers:
            with self.assertRaises(TypeError):
                n.validate(v)

        with self.assertRaises(ValueError):
            n.validate(float('nan'))

        self.assertEqual(repr(n), '<Numbers 0.1<=v<=3.5>')

    def test_failed_numbers(self):
        with self.assertRaises(TypeError):
            Numbers(1, 2, 3)

        with self.assertRaises(TypeError):
            Numbers(1, 1)  # min >= max

        for val in self.not_numbers:
            with self.assertRaises(TypeError):
                Numbers(max_value=val)

            with self.assertRaises(TypeError):
                Numbers(min_value=val)


class TestInts(TestCase):
    ints = [0, 1, 10, -1, 100, 1000000, int(-1e15), int(1e15),
            # warning: True==1 and False==0 - we *could* prohibit these, using
            # isinstance(v, bool)
            True, False,
            # np scalars
            np.int64(3)]
    not_ints = [0.1, -0.1, 1.0, 3.5, -2.3e6, 5.5e15, 1.34e-10, -2.5e-5,
                math.pi, math.e, '', None, float("nan"), float("inf"),
                -float("inf"), '1', [], {}, [1, 2], {1: 1}, b'good',
                AClass, AClass(), a_func]

    def test_unlimited(self):
        n = Ints()
        n.validate(n.valid_values[0])

        for v in self.ints:
            n.validate(v)

        for v in self.not_ints:
            with self.assertRaises(TypeError):
                n.validate(v)

    def test_min(self):
        for min_val in self.ints:
            n = Ints(min_value=min_val)
            for v in self.ints:
                if v >= min_val:
                    n.validate(v)
                else:
                    with self.assertRaises(ValueError):
                        n.validate(v)

        for v in self.not_ints:
            with self.assertRaises(TypeError):
                n.validate(v)

    def test_max(self):
        for max_val in self.ints:
            n = Ints(max_value=max_val)
            for v in self.ints:
                if v <= max_val:
                    n.validate(v)
                else:
                    with self.assertRaises(ValueError):
                        n.validate(v)

        for v in self.not_ints:
            with self.assertRaises(TypeError):
                n.validate(v)

    def test_range(self):
        n = Ints(0, 10)

        for v in self.ints:
            if 0 <= v <= 10:
                n.validate(v)
            else:
                with self.assertRaises(ValueError):
                    n.validate(v)

        for v in self.not_ints:
            with self.assertRaises(TypeError):
                n.validate(v)

        self.assertEqual(repr(n), '<Ints 0<=v<=10>')
        self.assertTrue(n.is_numeric)

    def test_failed_numbers(self):
        with self.assertRaises(TypeError):
            Ints(1, 2, 3)

        with self.assertRaises(TypeError):
            Ints(1, 1)  # min >= max

        for val in self.not_ints:
            with self.assertRaises((TypeError, OverflowError)):
                Ints(max_value=val)

            with self.assertRaises((TypeError, OverflowError)):
                Ints(min_value=val)


class TestPermissiveInts(TestCase):

    def test_close_to_ints(self):
        validator = PermissiveInts()
        validator.validate(validator.valid_values[0])

        a = 0
        b = 10
        values = np.linspace(a, b, b-a+1)
        for i in values:
            validator.validate(i)

    def test_bad_values(self):
        validator = PermissiveInts(0, 10)
        validator.validate(validator.valid_values[0])

        a = 0
        b = 10
        values = np.linspace(a, b, b-a+2)
        for j,i in enumerate(values):
            if j == 0 or j == 11:
                validator.validate(i)
            else:
                with self.assertRaises(TypeError):
                    validator.validate(i)


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
                with self.assertRaises((ValueError, TypeError)):
                    e.validate(v)

            self.assertEqual(repr(e), '<Enum: {}>'.format(repr(set(enum))))

            # Enum is never numeric, even if its members are all numbers
            # because the use of is_numeric is for sweeping
            self.assertFalse(e.is_numeric)

    def test_bad(self):
        for enum in self.not_enums:
            with self.assertRaises(TypeError):
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
                with self.assertRaises(ValueError):
                    n.validate(v)

            for v in self.not_multiples:
                with self.assertRaises(TypeError):
                    n.validate(v)

        for d in self.not_divisors:
            with self.assertRaises(TypeError):
                n = Multiples(divisor=d)

        n = Multiples(divisor=3, min_value=1, max_value=10)
        self.assertEqual(
            repr(n), '<Ints 1<=v<=10, Multiples of 3>')

    def test_valid_values(self):

        for d in self.divisors:
            n = Multiples(divisor=d)
            n.validate(n._valid_values[0])


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
                with self.assertRaises(ValueError):
                    val.validate(mult)

    # finally, a quick test that the precision is indeed setable
    def test_precision(self):
        pm_lax = PermissiveMultiples(35e-9, precision=3e-9)
        pm_lax.validate(72e-9)
        pm_strict = PermissiveMultiples(35e-9, precision=1e-10)
        with self.assertRaises(ValueError):
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
            with self.assertRaises(ValueError):
                m.validate(v)

        self.assertEqual(
            repr(m), '<MultiType: Strings 2<=len<=4, Ints 10<=v<=1000>')

        self.assertTrue(m.is_numeric)

        self.assertFalse(MultiType(Strings(), Enum(1, 2)).is_numeric)

    def test_bad(self):
        for args in [[], [1], [Strings(), True]]:
            with self.assertRaises(TypeError):
                MultiType(*args)

    def test_valid_values(self):
        m = MultiType(Strings(2, 4), Ints(10, 32))
        for val in m.valid_values:
            m.validate(val)


class TestArrays(TestCase):

    def test_type(self):
        m = Arrays(min_value=0.0, max_value=3.2, shape=(2, 2))
        for v in ['somestring', 4, 2, [[2, 0], [1, 2]]]:
            with self.assertRaises(TypeError):
                m.validate(v)

    def test_min_max(self):
        m = Arrays(min_value=-5, max_value=50, shape=(2, 2))
        v = np.array([[2, 0], [1, 2]])
        m.validate(v)
        v = 100*v
        with self.assertRaises(ValueError):
            m.validate(v)
        v = -1*v
        with self.assertRaises(ValueError):
            m.validate(v)

        m = Arrays(min_value=-5, shape=(2, 2))
        v = np.array([[2, 0], [1, 2]])
        m.validate(v*100)

    def test_shape(self):
        m = Arrays(min_value=-5, max_value=50, shape=(2, 2))
        v = np.array([[2, 0], [1, 2], [2, 3]])
        with self.assertRaises(ValueError):
            m.validate(v)

        # should pass if no shape specified
        m = Arrays(min_value=-5, max_value=50)
        m.validate(v)
        v = np.array([[2, 0], [1, 2]])
        m.validate(v)

    def test_valid_values(self):
        val = Arrays(min_value=-5, max_value=50, shape=(2, 2))
        val.validate(val.valid_values[0])


class TestLists(TestCase):

    def test_type(self):
        l = Lists()
        v1 = ['a', 'b', 5]
        l.validate(v1)

        v2 = 234
        with self.assertRaises(TypeError):
            l.validate(v2)

    def test_elt_vals(self):
        l = Lists(Ints(max_value=10))
        v1 = [0, 1, 5]
        l.validate(v1)

        v2 = [0, 1, 11]
        with self.assertRaises(ValueError):
            l.validate(v2)

    def test_valid_values(self):
        l = Lists(Ints(max_value=10))
        l.validate(l.valid_values[0])


class TestCallable(TestCase):

    def test_callable(self):
        c = Callable()

        def test_func():
            return True
        c.validate(test_func)
        test_int = 5
        with self.assertRaises(TypeError):
            c.validate(test_int)

    def test_valid_values(self):
        c = Callable()
        c.validate(c.valid_values[0])


class TestDict(TestCase):

    def test_dict(self):
        d = Dict()
        test_dict = {}
        d.validate(test_dict)
        test_int = 5
        with self.assertRaises(TypeError):
            d.validate(test_int)

    def test_valid_values(self):
        d = Dict()
        d.validate(d.valid_values[0])
