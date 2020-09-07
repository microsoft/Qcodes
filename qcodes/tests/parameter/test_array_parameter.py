from unittest import TestCase

from qcodes.instrument.parameter import ArrayParameter
from .conftest import blank_instruments, named_instrument


class SimpleArrayParam(ArrayParameter):
    def __init__(self, return_val, *args, **kwargs):
        self._return_val = return_val
        self._get_count = 0
        super().__init__(*args, **kwargs)

    def get_raw(self):
        self._get_count += 1
        return self._return_val


class SettableArray(SimpleArrayParam):
    # this is not allowed - just created to raise an error in the test below
    def set_raw(self, value):
        self.v = value


class TestArrayParameter(TestCase):
    def test_default_attributes(self):
        name = 'array_param'
        shape = (2, 3)
        p = SimpleArrayParam([[1, 2, 3], [4, 5, 6]], name, shape)

        self.assertEqual(p.name, name)
        self.assertEqual(p.shape, shape)

        self.assertEqual(p.label, name)
        self.assertEqual(p.unit, '')
        self.assertIsNone(p.setpoints)
        self.assertIsNone(p.setpoint_names)
        self.assertIsNone(p.setpoint_labels)

        self.assertEqual(str(p), name)

        self.assertEqual(p._get_count, 0)
        snap = p.snapshot(update=True)
        self.assertEqual(p._get_count, 0)
        snap_expected = {
            'name': name,
            'label': name,
            'unit': ''
        }
        for k, v in snap_expected.items():
            self.assertEqual(snap[k], v)
        self.assertNotIn('value', snap)
        self.assertNotIn('raw_value', snap)
        self.assertIsNone(snap['ts'])

        self.assertIn(name, p.__doc__)

    def test_explicit_attributes(self):
        name = 'tiny_array'
        shape = (2,)
        label = 'it takes two to tango'
        unit = 'steps'
        setpoints = [(0, 1)]
        setpoint_names = ['sp_index']
        setpoint_labels = ['Setpoint Label']
        docstring = 'Whats up Doc?'
        metadata = {'size': 2}
        p = SimpleArrayParam([6, 7], name, shape, label=label, unit=unit,
                             setpoints=setpoints,
                             setpoint_names=setpoint_names,
                             setpoint_labels=setpoint_labels,
                             docstring=docstring, snapshot_value=True,
                             metadata=metadata)

        self.assertEqual(p.name, name)
        self.assertEqual(p.shape, shape)
        self.assertEqual(p.label, label)
        self.assertEqual(p.unit, unit)
        self.assertEqual(p.setpoints, setpoints)
        self.assertEqual(p.setpoint_names, setpoint_names)
        self.assertEqual(p.setpoint_full_names, setpoint_names)
        self.assertEqual(p.setpoint_labels, setpoint_labels)

        self.assertEqual(p._get_count, 0)
        snap = p.snapshot(update=True)
        self.assertEqual(p._get_count, 1)
        snap_expected = {
            'name': name,
            'label': label,
            'unit': unit,
            'setpoint_names': setpoint_names,
            'setpoint_labels': setpoint_labels,
            'metadata': metadata,
            'value': [6, 7],
            'raw_value': [6, 7]
        }
        for k, v in snap_expected.items():
            self.assertEqual(snap[k], v)
        self.assertIsNotNone(snap['ts'])

        self.assertIn(name, p.__doc__)
        self.assertIn(docstring, p.__doc__)

    def test_has_set_get(self):
        name = 'array_param'
        shape = (3,)
        with self.assertRaises(AttributeError):
            ArrayParameter(name, shape)

        p = SimpleArrayParam([1, 2, 3], name, shape)

        self.assertTrue(hasattr(p, 'get'))
        self.assertTrue(p.gettable)
        self.assertFalse(hasattr(p, 'set'))
        self.assertFalse(p.settable)

        # Yet, it's possible to set the cached value
        p.cache.set([6, 7, 8])
        self.assertListEqual(p.get_latest(), [6, 7, 8])
        # However, due to the implementation of this ``SimpleArrayParam``
        # test parameter it's ``get`` call will return the originally passed
        # list
        self.assertListEqual(p.get(), [1, 2, 3])
        self.assertListEqual(p.get_latest(), [1, 2, 3])

        with self.assertRaises(AttributeError):
            SettableArray([1, 2, 3], name, shape)

    def test_full_name(self):
        # three cases where only name gets used for full_name
        for instrument in blank_instruments:
            p = SimpleArrayParam([6, 7], 'fred', (2,),
                                 setpoint_names=('barney',))
            p._instrument = instrument
            self.assertEqual(str(p), 'fred')
            self.assertEqual(p.setpoint_full_names, ('barney',))

        # and then an instrument that really has a name
        p = SimpleArrayParam([6, 7], 'wilma', (2,),
                             setpoint_names=('betty',))
        p._instrument = named_instrument
        self.assertEqual(str(p), 'astro_wilma')
        self.assertEqual(p.setpoint_full_names, ('astro_betty',))

        # and with a 2d parameter to test mixed setpoint_names
        p = SimpleArrayParam([[6, 7, 8], [1, 2, 3]], 'wilma', (3, 2),
                             setpoint_names=('betty', None))
        p._instrument = named_instrument
        self.assertEqual(p.setpoint_full_names, ('astro_betty', None))


    def test_constructor_errors(self):
        bad_constructors = [
            {'shape': [[3]]},  # not a depth-1 sequence
            {'shape': [3], 'setpoints': [1, 2, 3]},  # should be [[1, 2, 3]]
            {'shape': [3], 'setpoint_names': 'index'},  # should be ['index']
            {'shape': [3], 'setpoint_labels': 'the index'},  # ['the index']
            {'shape': [3], 'setpoint_names': [None, 'index2']}
        ]
        for kwargs in bad_constructors:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    SimpleArrayParam([1, 2, 3], 'p', **kwargs)
