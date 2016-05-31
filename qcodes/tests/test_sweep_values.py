from unittest import TestCase
from qcodes.instrument.parameter import StandardParameter, ManualParameter
from qcodes.instrument.sweep_values import SweepValues

from qcodes.utils.validators import Numbers


class TestSweepValues(TestCase):
    def setUp(self):
        self.c0 = ManualParameter('c0', vals=Numbers(-10, 10))
        self.c1 = ManualParameter('c1')

        self.getter = StandardParameter('c2', get_cmd=lambda: 42)

    def test_errors(self):
        c0 = self.c0

        # only complete 3-part slices are valid
        with self.assertRaises(TypeError):
            c0[1:2]  # For Int params this could be defined as step=1
        with self.assertRaises(TypeError):
            c0[:2:3]
        with self.assertRaises(TypeError):
            c0[1::3]
        with self.assertRaises(TypeError):
            c0[:]  # For Enum params we *could* define this one too...

        # fails if the parameter has no setter
        with self.assertRaises(TypeError):
            self.getter[0:0.1:0.01]

        # validates every step value against the parameter's Validator
        with self.assertRaises(ValueError):
            c0[5:15:1]
        with self.assertRaises(ValueError):
            c0[5.0:15.0:1.0]
        with self.assertRaises(ValueError):
            c0[-12]
        with self.assertRaises(ValueError):
            c0[-5, 12, 5]
        with self.assertRaises(ValueError):
            c0[-5, 12:8:1, 5]

        # cannot combine SweepValues for different parameters
        with self.assertRaises(TypeError):
            c0[0.1] + self.c1[0.2]

        # improper use of extend
        with self.assertRaises(TypeError):
            c0[0.1].extend(5)

        # SweepValue object has no getter, even if the parameter does
        with self.assertRaises(AttributeError):
            c0[0.1].get

    def test_valid(self):
        c0 = self.c0

        c0_sv = c0[1]
        # setter gets mapped
        self.assertEqual(c0_sv.set, c0.set)
        # normal sequence operations access values
        self.assertEqual(list(c0_sv), [1])
        self.assertEqual(c0_sv[0], 1)
        self.assertTrue(1 in c0_sv)
        self.assertFalse(2 in c0_sv)

        # in-place and copying addition
        c0_sv += c0[1.5:1.8:0.1]
        c0_sv2 = c0_sv + c0[2]
        self.assertEqual(list(c0_sv), [1, 1.5, 1.6, 1.7])
        self.assertEqual(list(c0_sv2), [1, 1.5, 1.6, 1.7, 2])

        # append and extend
        c0_sv3 = c0[2]
        # append only works with straight values
        c0_sv3.append(2.1)
        # extend can use another SweepValue, (even if it only has one value)
        c0_sv3.extend(c0[2.2])
        # extend can also take a sequence
        c0_sv3.extend([2.3])
        # as can addition
        c0_sv3 += [2.4]
        c0_sv4 = c0_sv3 + [2.5, 2.6]
        self.assertEqual(list(c0_sv3), [2, 2.1, 2.2, 2.3, 2.4])
        self.assertEqual(list(c0_sv4), [2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6])

        # len
        self.assertEqual(len(c0_sv3), 5)

        # in-place and copying reverse
        c0_sv.reverse()
        c0_sv5 = reversed(c0_sv)
        self.assertEqual(list(c0_sv), [1.7, 1.6, 1.5, 1])
        self.assertEqual(list(c0_sv5), [1, 1.5, 1.6, 1.7])

        # multi-key init, where first key is itself a list
        c0_sv6 = c0[[1, 3], 4]
        # copying
        c0_sv7 = c0_sv6.copy()
        self.assertEqual(list(c0_sv6), [1, 3, 4])
        self.assertEqual(list(c0_sv7), [1, 3, 4])
        self.assertFalse(c0_sv6 is c0_sv7)

    def test_base(self):
        p = ManualParameter('p')
        with self.assertRaises(NotImplementedError):
            iter(SweepValues(p))

    def test_snapshot(self):
        c0 = self.c0

        self.assertEqual(c0[0].snapshot(), {
            'parameter': c0.snapshot(),
            'values': [{'item': 0}]
        })

        self.assertEqual(c0[0:5:0.3].snapshot()['values'], [{
            'first': 0,
            'last': 4.8,
            'num': 17,
            'type': 'linear'
        }])

        sv = c0.sweep(start=2, stop=4, num=5)
        self.assertEqual(sv.snapshot()['values'], [{
            'first': 2,
            'last': 4,
            'num': 5,
            'type': 'linear'
        }])

        # mixture of bare items, nested lists, and slices
        sv = c0[1, 7, 3.2, [1, 2, 3], 6:9:1, -4.5, 5.3]
        self.assertEqual(sv.snapshot()['values'], [{
            'first': 1,
            'last': 5.3,
            'min': -4.5,
            'max': 8,
            'num': 11,
            'type': 'sequence'
            }])

        self.assertEqual((c0[0] + c0[1]).snapshot()['values'], [
            {'item': 0},
            {'item': 1}
            ])

        self.assertEqual((c0[0:3:1] + c0[4, 6, 9]).snapshot()['values'], [
            {'first': 0, 'last': 2, 'num': 3, 'type': 'linear'},
            {'first': 4, 'last': 9, 'min': 4, 'max': 9, 'num': 3,
             'type': 'sequence'}
            ])

    def test_repr(self):
        sv = self.c0[0]
        self.assertEqual(repr(sv),
                         '<qcodes.instrument.sweep_values.SweepFixedValues: '
                         'c0 at {}>'.format(id(sv)))
