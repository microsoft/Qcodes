"""
Test suite for parameter
"""
from collections import namedtuple
from unittest import TestCase

from qcodes import Function
from qcodes.instrument.parameter import (
    Parameter, ArrayParameter, MultiParameter,
    ManualParameter, StandardParameter, InstrumentRefParameter)
from qcodes.utils.helpers import LogCapture
from qcodes.utils.validators import Numbers
from qcodes.tests.instrument_mocks import DummyInstrument


class GettableParam(Parameter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._get_count = 0

    def get(self):
        self._get_count += 1
        self._save_val(42)
        return 42


class SimpleManualParam(Parameter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._save_val(0)
        self._v = 0

    def get(self):
        return self._v

    def set(self, v):
        self._save_val(v)
        self._v = v


class SettableParam(Parameter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._save_val(0)
        self._v = 0

    def set(self, v):
        self._save_val(v)
        self._v = v


blank_instruments = (
    None,  # no instrument at all
    namedtuple('noname', '')(),  # no .name
    namedtuple('blank', 'name')('')  # blank .name
)
named_instrument = namedtuple('yesname', 'name')('astro')


class TestParameter(TestCase):
    def test_no_name(self):
        with self.assertRaises(TypeError):
            GettableParam()

    def test_default_attributes(self):
        # Test the default attributes, providing only a name
        name = 'repetitions'
        p = GettableParam(name)
        self.assertEqual(p.name, name)
        self.assertEqual(p.label, name)
        self.assertEqual(p.unit, '')
        self.assertEqual(p.full_name, name)

        # default validator is all numbers
        p.validate(-1000)
        with self.assertRaises(TypeError):
            p.validate('not a number')

        # docstring exists, even without providing one explicitly
        self.assertIn(name, p.__doc__)

        # test snapshot_get by looking at _get_count
        self.assertEqual(p._get_count, 0)
        snap = p.snapshot(update=True)
        self.assertEqual(p._get_count, 1)
        snap_expected = {
            'name': name,
            'label': name,
            'unit': '',
            'value': 42,
            'vals': repr(Numbers())
        }
        for k, v in snap_expected.items():
            self.assertEqual(snap[k], v)

    def test_explicit_attributes(self):
        # Test the explicit attributes, providing everything we can
        name = 'volt'
        label = 'Voltage'
        unit = 'V'
        docstring = 'DOCS!'
        metadata = {'gain': 100}
        p = GettableParam(name, label=label, unit=unit,
                          vals=Numbers(5, 10), docstring=docstring,
                          snapshot_get=False, metadata=metadata)

        self.assertEqual(p.name, name)
        self.assertEqual(p.label, label)
        self.assertEqual(p.unit, unit)
        self.assertEqual(p.full_name, name)

        with self.assertRaises(ValueError):
            p.validate(-1000)
        p.validate(6)
        with self.assertRaises(TypeError):
            p.validate('not a number')

        self.assertIn(name, p.__doc__)
        self.assertIn(docstring, p.__doc__)

        # test snapshot_get by looking at _get_count
        self.assertEqual(p._get_count, 0)
        snap = p.snapshot(update=True)
        self.assertEqual(p._get_count, 0)
        snap_expected = {
            'name': name,
            'label': label,
            'unit': unit,
            'vals': repr(Numbers(5, 10)),
            'metadata': metadata
        }
        for k, v in snap_expected.items():
            self.assertEqual(snap[k], v)

        # attributes only available in MultiParameter
        for attr in ['names', 'labels', 'setpoints', 'setpoint_names',
                     'setpoint_labels', 'full_names']:
            self.assertFalse(hasattr(p, attr), attr)

    def test_units(self):
        with LogCapture() as logs:
            p = GettableParam('p', units='V')

        self.assertIn('deprecated', logs.value)
        self.assertEqual(p.unit, 'V')

        with LogCapture() as logs:
            self.assertEqual(p.units, 'V')

        self.assertIn('deprecated', logs.value)

        with LogCapture() as logs:
            p = GettableParam('p', unit='Tesla', units='Gauss')

        self.assertIn('deprecated', logs.value)
        self.assertEqual(p.unit, 'Tesla')

        with LogCapture() as logs:
            self.assertEqual(p.units, 'Tesla')

        self.assertIn('deprecated', logs.value)

    def test_repr(self):
        for i in [0, "foo", "", "fåil"]:
            with self.subTest(i=i):
                param = GettableParam(name=i)
                s = param.__repr__()
                st = '<{}.{}: {} at {}>'.format(
                    param.__module__, param.__class__.__name__,
                    param.name, id(param))
                self.assertEqual(s, st)

    def test_has_set_get(self):
        # you can't instantiate a Parameter directly anymore, only a subclass,
        # because you need a get or a set method.
        with self.assertRaises(AttributeError):
            Parameter('no_get_or_set')

        gp = GettableParam('1')
        self.assertTrue(gp.has_get)
        self.assertFalse(gp.has_set)
        with self.assertRaises(NotImplementedError):
            gp(1)

        sp = SettableParam('2')
        self.assertFalse(sp.has_get)
        self.assertTrue(sp.has_set)
        with self.assertRaises(NotImplementedError):
            sp()

        sgp = SimpleManualParam('3')
        self.assertTrue(sgp.has_get)
        self.assertTrue(sgp.has_set)
        sgp(22)
        self.assertEqual(sgp(), 22)

    def test_full_name(self):
        # three cases where only name gets used for full_name
        for instrument in blank_instruments:
            p = GettableParam(name='fred')
            p._instrument = instrument
            self.assertEqual(p.full_name, 'fred')

        # and finally an instrument that really has a name
        p = GettableParam(name='wilma')
        p._instrument = named_instrument
        self.assertEqual(p.full_name, 'astro_wilma')

    def test_bad_validator(self):
        with self.assertRaises(TypeError):
            GettableParam('p', vals=[1, 2, 3])


class SimpleArrayParam(ArrayParameter):
    def __init__(self, return_val, *args, **kwargs):
        self._return_val = return_val
        self._get_count = 0
        super().__init__(*args, **kwargs)

    def get(self):
        self._get_count += 1
        self._save_val(self._return_val)
        return self._return_val


class SettableArray(SimpleArrayParam):
    # this is not allowed - just created to raise an error in the test below
    def set(self, v):
        self.v = v


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

        self.assertEqual(p.full_name, name)

        self.assertEqual(p._get_count, 0)
        snap = p.snapshot(update=True)
        self.assertEqual(p._get_count, 1)
        snap_expected = {
            'name': name,
            'label': name,
            'unit': '',
            'value': [[1, 2, 3], [4, 5, 6]]
        }
        for k, v in snap_expected.items():
            self.assertEqual(snap[k], v)

        self.assertIn(name, p.__doc__)

    def test_explicit_attrbutes(self):
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
                             docstring=docstring, snapshot_get=False,
                             metadata=metadata)

        self.assertEqual(p.name, name)
        self.assertEqual(p.shape, shape)
        self.assertEqual(p.label, label)
        self.assertEqual(p.unit, unit)
        self.assertEqual(p.setpoints, setpoints)
        self.assertEqual(p.setpoint_names, setpoint_names)
        self.assertEqual(p.setpoint_labels, setpoint_labels)

        self.assertEqual(p._get_count, 0)
        snap = p.snapshot(update=True)
        self.assertEqual(p._get_count, 0)
        snap_expected = {
            'name': name,
            'label': label,
            'unit': unit,
            'setpoint_names': setpoint_names,
            'setpoint_labels': setpoint_labels,
            'metadata': metadata
        }
        for k, v in snap_expected.items():
            self.assertEqual(snap[k], v)

        self.assertIn(name, p.__doc__)
        self.assertIn(docstring, p.__doc__)

    def test_units(self):
        with LogCapture() as logs:
            p = SimpleArrayParam([6, 7], 'p', (2,), units='V')

        self.assertIn('deprecated', logs.value)
        self.assertEqual(p.unit, 'V')

        with LogCapture() as logs:
            self.assertEqual(p.units, 'V')

        self.assertIn('deprecated', logs.value)

        with LogCapture() as logs:
            p = SimpleArrayParam([6, 7], 'p', (2,),
                                 unit='Tesla', units='Gauss')

        self.assertIn('deprecated', logs.value)
        self.assertEqual(p.unit, 'Tesla')

        with LogCapture() as logs:
            self.assertEqual(p.units, 'Tesla')

        self.assertIn('deprecated', logs.value)

    def test_has_set_get(self):
        name = 'array_param'
        shape = (3,)
        with self.assertRaises(AttributeError):
            ArrayParameter(name, shape)

        p = SimpleArrayParam([1, 2, 3], name, shape)

        self.assertTrue(p.has_get)
        self.assertFalse(p.has_set)

        with self.assertRaises(AttributeError):
            SettableArray([1, 2, 3], name, shape)

    def test_full_name(self):
        # three cases where only name gets used for full_name
        for instrument in blank_instruments:
            p = SimpleArrayParam([6, 7], 'fred', (2,))
            p._instrument = instrument
            self.assertEqual(p.full_name, 'fred')

        # and finally an instrument that really has a name
        p = SimpleArrayParam([6, 7], 'wilma', (2,))
        p._instrument = named_instrument
        self.assertEqual(p.full_name, 'astro_wilma')

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


class SimpleMultiParam(MultiParameter):
    def __init__(self, return_val, *args, **kwargs):
        self._return_val = return_val
        self._get_count = 0
        super().__init__(*args, **kwargs)

    def get(self):
        self._get_count += 1
        self._save_val(self._return_val)
        return self._return_val


class SettableMulti(SimpleMultiParam):
    # this is not fully suported - just created to raise a warning in the test below.
    # We test that the warning is raised
    def set(self, v):
        print("Calling set")
        self.v = v


class TestMultiParameter(TestCase):
    def test_default_attributes(self):
        name = 'mixed_dimensions'
        names = ['0D', '1D', '2D']
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

        self.assertEqual(p.full_name, name)

        self.assertEqual(p._get_count, 0)
        snap = p.snapshot(update=True)
        self.assertEqual(p._get_count, 1)
        snap_expected = {
            'name': name,
            'names': names,
            'labels': names,
            'units': [''] * 3,
            'value': [0, [1, 2, 3], [[4, 5], [6, 7]]]
        }
        for k, v in snap_expected.items():
            self.assertEqual(snap[k], v)

        self.assertIn(name, p.__doc__)

        # only in simple parameters
        self.assertFalse(hasattr(p, 'label'))
        self.assertFalse(hasattr(p, 'unit'))

    def test_explicit_attributes(self):
        name = 'mixed_dimensions'
        names = ['0D', '1D', '2D']
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
                             docstring=docstring, snapshot_get=False,
                             metadata=metadata)

        self.assertEqual(p.name, name)
        self.assertEqual(p.names, names)
        self.assertEqual(p.shapes, shapes)

        self.assertEqual(p.labels, labels)
        self.assertEqual(p.units, units)
        self.assertEqual(p.setpoints, setpoints)
        self.assertEqual(p.setpoint_names, setpoint_names)
        self.assertEqual(p.setpoint_labels, setpoint_labels)

        self.assertEqual(p._get_count, 0)
        snap = p.snapshot(update=True)
        self.assertEqual(p._get_count, 0)
        snap_expected = {
            'name': name,
            'names': names,
            'labels': labels,
            'units': units,
            'setpoint_names': setpoint_names,
            'setpoint_labels': setpoint_labels,
            'metadata': metadata
        }
        for k, v in snap_expected.items():
            self.assertEqual(snap[k], v)

        self.assertIn(name, p.__doc__)
        self.assertIn(docstring, p.__doc__)

    def test_has_set_get(self):
        name = 'mixed_dimensions'
        names = ['0D', '1D', '2D']
        shapes = ((), (3,), (2, 2))
        with self.assertRaises(AttributeError):
            MultiParameter(name, names, shapes)

        p = SimpleMultiParam([0, [1, 2, 3], [[4, 5], [6, 7]]],
                             name, names, shapes)

        self.assertTrue(p.has_get)
        self.assertFalse(p.has_set)
        # We allow creation of Multiparameters with set to support
        # instruments that already make use of them.
        with self.assertWarns(UserWarning):
            SettableMulti([0, [1, 2, 3], [[4, 5], [6, 7]]],
                          name, names, shapes)

    def test_full_name_s(self):
        name = 'mixed_dimensions'
        names = ['0D', '1D', '2D']
        shapes = ((), (3,), (2, 2))

        # three cases where only name gets used for full_name
        for instrument in blank_instruments:
            p = SimpleMultiParam([0, [1, 2, 3], [[4, 5], [6, 7]]],
                                 name, names, shapes)
            p._instrument = instrument
            self.assertEqual(p.full_name, name)

            self.assertEqual(p.full_names, names)

        # and finally an instrument that really has a name
        p = SimpleMultiParam([0, [1, 2, 3], [[4, 5], [6, 7]]],
                             name, names, shapes)
        p._instrument = named_instrument
        self.assertEqual(p.full_name, 'astro_mixed_dimensions')

        self.assertEqual(p.full_names, ['astro_0D', 'astro_1D', 'astro_2D'])

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


class TestManualParameter(TestCase):

    def test_bare_function(self):
        # not a use case we want to promote, but it's there...
        p = ManualParameter('test')

        def doubler(x):
            p.set(x * 2)

        f = Function('f', call_cmd=doubler, args=[Numbers(-10, 10)])

        f(4)
        self.assertEqual(p.get(), 8)
        with self.assertRaises(ValueError):
            f(20)


class TestStandardParam(TestCase):
    def set_p(self, val):
        self._p = val

    def set_p_prefixed(self, val):
        self._p = 'PVAL: {:d}'.format(val)

    def strip_prefix(self, val):
        return int(val[6:])

    def get_p(self):
        return self._p

    def parse_set_p(self, val):
        return '{:d}'.format(val)

    def test_param_cmd_with_parsing(self):
        p = StandardParameter('p_int', get_cmd=self.get_p, get_parser=int,
                              set_cmd=self.set_p, set_parser=self.parse_set_p)

        p(5)
        self.assertEqual(self._p, '5')
        self.assertEqual(p(), 5)

    def test_settable(self):
        p = StandardParameter('p', set_cmd=self.set_p)

        p(10)
        self.assertEqual(self._p, 10)
        with self.assertRaises(NotImplementedError):
            p()
        with self.assertRaises(NotImplementedError):
            p.get()

        self.assertTrue(p.has_set)
        self.assertFalse(p.has_get)

    def test_gettable(self):
        p = StandardParameter('p', get_cmd=self.get_p)
        self._p = 21

        self.assertEqual(p(), 21)
        self.assertEqual(p.get(), 21)

        with self.assertRaises(NotImplementedError):
            p(10)
        with self.assertRaises(NotImplementedError):
            p.set(10)

        self.assertTrue(p.has_get)
        self.assertFalse(p.has_set)

    def test_val_mapping_basic(self):
        p = StandardParameter('p', set_cmd=self.set_p, get_cmd=self.get_p,
                              val_mapping={'off': 0, 'on': 1})

        p('off')
        self.assertEqual(self._p, 0)
        self.assertEqual(p(), 'off')

        self._p = 1
        self.assertEqual(p(), 'on')

        # implicit mapping to ints
        self._p = '0'
        self.assertEqual(p(), 'off')

        # unrecognized response
        self._p = 2
        with self.assertRaises(KeyError):
            p()

    def test_val_mapping_with_parsers(self):
        # you can't use set_parser with val_mapping... just too much
        # indirection since you also have set_cmd
        with self.assertRaises(TypeError):
            StandardParameter('p', set_cmd=self.set_p, get_cmd=self.get_p,
                              val_mapping={'off': 0, 'on': 1},
                              set_parser=self.parse_set_p)

        # but you *can* use get_parser with val_mapping
        p = StandardParameter('p', set_cmd=self.set_p_prefixed,
                              get_cmd=self.get_p, get_parser=self.strip_prefix,
                              val_mapping={'off': 0, 'on': 1})

        p('off')
        self.assertEqual(self._p, 'PVAL: 0')
        self.assertEqual(p(), 'off')

        self._p = 'PVAL: 1'
        self.assertEqual(p(), 'on')


class TestInstrumentRefParameter(TestCase):
    def test_get_instr(self):
        a = DummyInstrument('dummy_holder')
        d = DummyInstrument('dummy')
        a.add_parameter('test', parameter_class=InstrumentRefParameter)

        a.test.set(d.name)

        self.assertEqual(a.test.get(), d.name)
        self.assertEqual(a.test.get_instr(), d)
