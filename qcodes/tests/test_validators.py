from unittest import TestCase
import math

from qcodes.utils.validators import (Validator, Anything, Bool, Strings,
                                     Numbers, Ints, Enum, MultiType)


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
        # nor can you call is_valid without overriding it in a subclass
        b = self.BrokenValidator()
        with self.assertRaises(NotImplementedError):
            b.is_valid(0)


class TestAnything(TestCase):
    def test_real_anything(self):
        a = Anything()
        for v in [None, 0, 1, 0.0, 1.2, '', 'hi!', [1, 2, 3], [],
                  {'a': 1, 'b': 2}, {}, set([1, 2, 3]), a, range(10),
                  True, False, float("nan"), float("inf"), b'good',
                  AClass, AClass(), a_func]:
            self.assertTrue(a.is_valid(v))

        self.assertEqual(repr(a), '<Anything>')

    def test_failed_anything(self):
        with self.assertRaises(TypeError):
            Anything(1)

        with self.assertRaises(TypeError):
            Anything(values=[1, 2, 3])


class TestBool(TestCase):
    bools = [True, False]
    not_bools = [0, 1, 10, -1, 100, 1000000, int(-1e15), int(1e15),
                 0.1, -0.1, 1.0, 3.5, -2.3e6, 5.5e15, 1.34e-10, -2.5e-5,
                 math.pi, math.e, '', None, float("nan"), float("inf"),
                 -float("inf"), '1', [], {}, [1, 2], {1: 1}, b'good',
                 AClass, AClass(), a_func]

    def test_bool(self):
        b = Bool()

        for v in self.bools:
            self.assertTrue(b.is_valid(v))

        for v in self.not_bools:
            self.assertFalse(b.is_valid(v))

        self.assertEqual(repr(b), '<Boolean>')


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
            self.assertTrue(s.is_valid(v))

        for v in self.not_strings:
            self.assertFalse(s.is_valid(v))

        self.assertEqual(repr(s), '<Strings>')

    def test_min(self):
        for min_len in [0, 1, 5, 10, 100]:
            s = Strings(min_length=min_len)
            for v in self.strings:
                self.assertEqual(s.is_valid(v), len(v) >= min_len)

        for v in self.not_strings:
            self.assertFalse(s.is_valid(v))

        self.assertEqual(repr(s), '<Strings len>=100>')

    def test_max(self):
        for max_len in [1, 5, 10, 100]:
            s = Strings(max_length=max_len)
            for v in self.strings:
                self.assertEqual(s.is_valid(v), len(v) <= max_len)

        for v in self.not_strings:
            self.assertFalse(s.is_valid(v))

        self.assertEqual(repr(s), '<Strings len<=100>')

    def test_range(self):
        s = Strings(1, 10)

        for v in self.strings:
            self.assertEqual(s.is_valid(v), 1 <= len(v) <= 10)

        for v in self.not_strings:
            self.assertFalse(s.is_valid(v))

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


class TestNumbers(TestCase):
    numbers = [0, 1, -1, 0.1, -0.1, 100, 1000000, 1.0, 3.5, -2.3e6, 5.5e15,
               1.34e-10, -2.5e-5, math.pi, math.e,
               # warning: True==1 and False==0
               True, False,
               # warning: +/- inf are allowed if max & min are not specified!
               -float("inf"), float("inf")]
    not_numbers = ['', None, float("nan"), '1', [], {}, [1, 2], {1: 1},
                   b'good', AClass, AClass(), a_func]

    def test_unlimited(self):
        n = Numbers()

        for v in self.numbers:
            self.assertTrue(n.is_valid(v))

        for v in self.not_numbers:
            self.assertFalse(n.is_valid(v))

    def test_min(self):
        for min_val in [-1e20, -1, -0.1, 0, 0.1, 10]:
            n = Numbers(min_value=min_val)
            for v in self.numbers:
                self.assertEqual(n.is_valid(v), v >= min_val)

        for v in self.not_numbers:
            self.assertFalse(n.is_valid(v))

    def test_max(self):
        for max_val in [-1e20, -1, -0.1, 0, 0.1, 10]:
            n = Numbers(max_value=max_val)
            for v in self.numbers:
                self.assertEqual(n.is_valid(v), v <= max_val)

        for v in self.not_numbers:
            self.assertFalse(n.is_valid(v))

    def test_range(self):
        n = Numbers(0.1, 3.5)

        for v in self.numbers:
            self.assertEqual(n.is_valid(v), 0.1 <= v <= 3.5)

        for v in self.not_numbers:
            self.assertFalse(n.is_valid(v))

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
            True, False]
    not_ints = [0.1, -0.1, 1.0, 3.5, -2.3e6, 5.5e15, 1.34e-10, -2.5e-5,
                math.pi, math.e, '', None, float("nan"), float("inf"),
                -float("inf"), '1', [], {}, [1, 2], {1: 1}, b'good',
                AClass, AClass(), a_func]

    def test_unlimited(self):
        n = Ints()

        for v in self.ints:
            self.assertTrue(n.is_valid(v))

        for v in self.not_ints:
            self.assertFalse(n.is_valid(v))

    def test_min(self):
        for min_val in self.ints:
            n = Ints(min_value=min_val)
            for v in self.ints:
                self.assertEqual(n.is_valid(v), v >= min_val)

        for v in self.not_ints:
            self.assertFalse(n.is_valid(v))

    def test_max(self):
        for max_val in self.ints:
            n = Ints(max_value=max_val)
            for v in self.ints:
                self.assertEqual(n.is_valid(v), v <= max_val)

        for v in self.not_ints:
            self.assertFalse(n.is_valid(v))

    def test_range(self):
        n = Ints(0, 10)

        for v in self.ints:
            self.assertEqual(n.is_valid(v), 0 <= v <= 10)

        for v in self.not_ints:
            self.assertFalse(n.is_valid(v))

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
                self.assertTrue(e.is_valid(v))

            for v in [22, 'bad data', [44, 55]]:
                self.assertFalse(e.is_valid(v))

            self.assertEqual(repr(e), '<Enum: {}>'.format(repr(set(enum))))

            # Enum is never numeric, even if its members are all numbers
            # because the use of is_numeric is for sweeping
            self.assertFalse(e.is_numeric)

    def test_bad(self):
        for enum in self.not_enums:
            with self.assertRaises(TypeError):
                Enum(*enum)


class TestMultiType(TestCase):
    def test_good(self):
        m = MultiType(Strings(2, 4), Ints(10, 1000))

        for v in [10, 11, 123, 1000, 'aa', 'mop', 'FRED']:
            self.assertTrue(m.is_valid(v))

        for v in [9, 1001, 'Q', 'Qcode', None, 100.0, b'nice', [], {},
                  a_func, AClass, AClass(), True, False]:
            self.assertFalse(m.is_valid(v))

        self.assertEqual(
            repr(m), '<MultiType: Strings 2<=len<=4, Ints 10<=v<=1000>')

        self.assertTrue(m.is_numeric)

        self.assertFalse(MultiType(Strings(), Enum(1, 2)).is_numeric)

    def test_bad(self):
        for args in [[], [1], [Strings(), True]]:
            with self.assertRaises(TypeError):
                MultiType(*args)
