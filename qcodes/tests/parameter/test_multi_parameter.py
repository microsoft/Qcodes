from unittest import TestCase

from qcodes.instrument.parameter import MultiParameter
from .conftest import named_instrument, blank_instruments


class SimpleMultiParam(MultiParameter):
    def __init__(self, return_val, *args, **kwargs):
        self._return_val = return_val
        self._get_count = 0
        super().__init__(*args, **kwargs)

    def get_raw(self):
        self._get_count += 1
        return self._return_val


class SettableMulti(SimpleMultiParam):
    def set_raw(self, value):
        print("Calling set")
        self._return_val = value


class TestMultiParameter(TestCase):
    def test_default_attributes(self):
        name = 'mixed_dimensions'
        names = ('0D', '1D', '2D')
        shapes = ((), (3,), (2, 2))
        p = SimpleMultiParam([0, [1, 2, 3], [[4, 5], [6, 7]]],
                             name, names, shapes)

        self.assertEqual(p.name, name)
        self.assertEqual(p.names, names)
        self.assertEqual(p.shapes, shapes)

        self.assertEqual(p.labels, names)
        self.assertEqual(p.units, [''] * 3)
        self.assertIsNone(p.setpoints)
        self.assertIsNone(p.setpoint_names)
        self.assertIsNone(p.setpoint_labels)

        self.assertEqual(str(p), name)

        self.assertEqual(p._get_count, 0)
        snap = p.snapshot(update=True)
        self.assertEqual(p._get_count, 0)
        snap_expected = {
            'name': name,
            'names': names,
            'labels': names,
            'units': [''] * 3,
            'ts': None
        }
        for k, v in snap_expected.items():
            self.assertEqual(snap[k], v)
        self.assertNotIn('value', snap)
        self.assertNotIn('raw_value', snap)

        self.assertIn(name, p.__doc__)

        # only in simple parameters
        self.assertFalse(hasattr(p, 'label'))
        self.assertFalse(hasattr(p, 'unit'))

    def test_explicit_attributes(self):
        name = 'mixed_dimensions'
        names = ('0D', '1D', '2D')
        shapes = ((), (3,), (2, 2))
        labels = ['scalar', 'vector', 'matrix']
        units = ['V', 'A', 'W']
        setpoints = [(), ((4, 5, 6),), ((7, 8), None)]
        setpoint_names = [(), ('sp1',), ('sp2', None)]
        setpoint_labels = [(), ('setpoint1',), ('setpoint2', None)]
        docstring = 'DOCS??'
        metadata = {'sizes': [1, 3, 4]}
        p = SimpleMultiParam([0, [1, 2, 3], [[4, 5], [6, 7]]],
                             name, names, shapes, labels=labels, units=units,
                             setpoints=setpoints,
                             setpoint_names=setpoint_names,
                             setpoint_labels=setpoint_labels,
                             docstring=docstring, snapshot_value=True,
                             metadata=metadata)

        self.assertEqual(p.name, name)
        self.assertEqual(p.names, names)
        self.assertEqual(p.shapes, shapes)

        self.assertEqual(p.labels, labels)
        self.assertEqual(p.units, units)
        self.assertEqual(p.setpoints, setpoints)
        self.assertEqual(p.setpoint_names, setpoint_names)
        # as the parameter is not attached to an instrument the full names are
        # equivalent to the setpoint_names
        self.assertEqual(p.setpoint_full_names, setpoint_names)
        self.assertEqual(p.setpoint_labels, setpoint_labels)

        self.assertEqual(p._get_count, 0)
        snap = p.snapshot(update=True)
        self.assertEqual(p._get_count, 1)
        snap_expected = {
            'name': name,
            'names': names,
            'labels': labels,
            'units': units,
            'setpoint_names': setpoint_names,
            'setpoint_labels': setpoint_labels,
            'metadata': metadata,
            'value': [0, [1, 2, 3], [[4, 5], [6, 7]]],
            'raw_value': [0, [1, 2, 3], [[4, 5], [6, 7]]]
        }
        for k, v in snap_expected.items():
            self.assertEqual(snap[k], v)
        self.assertIsNotNone(snap['ts'])

        self.assertIn(name, p.__doc__)
        self.assertIn(docstring, p.__doc__)

    def test_has_set_get(self):
        name = 'mixed_dimensions'
        names = ['0D', '1D', '2D']
        shapes = ((), (3,), (2, 2))
        with self.assertRaises(AttributeError):
            MultiParameter(name, names, shapes)

        original_value = [0, [1, 2, 3], [[4, 5], [6, 7]]]
        p = SimpleMultiParam(original_value, name, names, shapes)

        self.assertTrue(hasattr(p, 'get'))
        self.assertTrue(p.gettable)
        self.assertFalse(hasattr(p, 'set'))
        self.assertFalse(p.settable)
        # Ensure that ``cache.set`` works
        new_cache = [10, [10, 20, 30], [[40, 50], [60, 70]]]
        p.cache.set(new_cache)
        self.assertListEqual(p.get_latest(), new_cache)
        # However, due to the implementation of this ``SimpleMultiParam``
        # test parameter it's ``get`` call will return the originally passed
        # list
        self.assertListEqual(p.get(), original_value)
        self.assertListEqual(p.get_latest(), original_value)

        # We allow creation of Multiparameters with set to support
        # instruments that already make use of them.
        p = SettableMulti([0, [1, 2, 3], [[4, 5], [6, 7]]], name, names, shapes)
        self.assertTrue(hasattr(p, 'get'))
        self.assertTrue(p.gettable)
        self.assertTrue(hasattr(p, 'set'))
        self.assertTrue(p.settable)
        value_to_set = [2, [1, 5, 2], [[8, 2], [4, 9]]]
        p.set(value_to_set)
        assert p.get() == value_to_set
        # Also, ``cache.set`` works as expected
        p.cache.set(new_cache)
        assert p.get_latest() == new_cache
        assert p.get() == value_to_set

    def test_full_name_s(self):
        name = 'mixed_dimensions'
        names = ('0D', '1D', '2D')
        setpoint_names = ((),
                          ('setpoints_1D',),
                          ('setpoints_2D_1',
                           None))
        shapes = ((), (3,), (2, 2))

        # three cases where only name gets used for full_name
        for instrument in blank_instruments:
            p = SimpleMultiParam([0, [1, 2, 3], [[4, 5], [6, 7]]],
                                 name, names, shapes,
                                 setpoint_names=setpoint_names)
            p._instrument = instrument
            self.assertEqual(str(p), name)
            self.assertEqual(p.full_names, names)
            self.assertEqual(p.setpoint_full_names,
                             ((), ('setpoints_1D',), ('setpoints_2D_1', None)))

        # and finally an instrument that really has a name
        p = SimpleMultiParam([0, [1, 2, 3], [[4, 5], [6, 7]]],
                             name, names, shapes, setpoint_names=setpoint_names)
        p._instrument = named_instrument
        self.assertEqual(str(p), 'astro_mixed_dimensions')

        self.assertEqual(p.full_names, ('astro_0D', 'astro_1D', 'astro_2D'))
        self.assertEqual(p.setpoint_full_names,
                         ((), ('astro_setpoints_1D',),
                          ('astro_setpoints_2D_1', None)))

    def test_constructor_errors(self):
        bad_constructors = [
            {'names': 'a', 'shapes': ((3,), ())},
            {'names': ('a', 'b'), 'shapes': (3, 2)},
            {'names': ('a', 'b'), 'shapes': ((3,), ()),
             'setpoints': [(1, 2, 3), ()]},
            {'names': ('a', 'b'), 'shapes': ((3,), ()),
             'setpoint_names': (None, ('index',))},
            {'names': ('a', 'b'), 'shapes': ((3,), ()),
             'setpoint_labels': (None, None, None)}
        ]
        for kwargs in bad_constructors:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    SimpleMultiParam([1, 2, 3], 'p', **kwargs)
