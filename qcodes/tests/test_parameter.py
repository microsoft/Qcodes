"""
Test suite for parameter
"""
from collections import namedtuple
from unittest import TestCase
from time import sleep
import weakref
import gc
from copy import copy, deepcopy
import logging
import pickle

import numpy as np
from hypothesis import given
import hypothesis.strategies as hst
from qcodes import Function
from qcodes.instrument.parameter import (
    Parameter, ArrayParameter, MultiParameter,
    InstrumentRefParameter)
import qcodes.utils.validators as vals
from qcodes.tests.instrument_mocks import DummyInstrument


class GettableParam(Parameter):
    """ Parameter that keeps track of number of get operations"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._get_count = 0

    def get_raw(self):
        self._get_count += 1
        self._save_val(42)
        return 42


class BookkeepingValidator(vals.Validator):
    """
    Validator that keeps track of what it validates
    """
    def __init__(self, min_value=-float("inf"), max_value=float("inf")):
        self.values_validated = []

    def validate(self, value, context=''):
        self.values_validated.append(value)

    is_numeric = True


blank_instruments = (
    None,  # no instrument at all
    namedtuple('noname', '')(),  # no .name
    namedtuple('blank', 'name')('')  # blank .name
)
named_instrument = namedtuple('yesname', 'name')('astro')


class MemoryParameter(Parameter):
    def __init__(self, get_cmd=None, **kwargs):
        self.set_values = []
        self.get_values = []
        super().__init__(set_cmd=self.add_set_value,
                         get_cmd=self.create_get_func(get_cmd), **kwargs)

    def add_set_value(self, value):
        self.set_values.append(value)

    def create_get_func(self, func):
        def get_func():
            if func is not None:
                val = func()
            else:
                val = self._latest['raw_value']
            self.get_values.append(val)
            return val
        return get_func


class TestParameter(TestCase):
    def test_default_attributes(self):
        # Test the default attributes, providing only a name
        name = 'repetitions'
        p = GettableParam(name, vals=vals.Numbers())
        self.assertEqual(p.name, name)
        self.assertEqual(p.label, name.capitalize())
        self.assertEqual(p.unit, '')
        self.assertEqual(str(p), name)

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
            'label': name.capitalize(),
            'unit': '',
            'value': 42,
            'vals': repr(vals.Numbers())
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
                          vals=vals.Numbers(5, 10), docstring=docstring,
                          snapshot_get=False, metadata=metadata)

        self.assertEqual(p.name, name)
        self.assertEqual(p.label, label)
        self.assertEqual(p.unit, unit)
        self.assertEqual(str(p), name)

        with self.assertRaises(ValueError):
            p.validate(-1000)
        p.validate(6)
        with self.assertRaises(TypeError):
            p.validate('not a number')

        self.assertIn(name, p.__doc__)
        self.assertIn(docstring, p.__doc__)

        # test snapshot_get by looking at _get_count
        self.assertEqual(p._get_count, 0)
        # Snapshot should not perform get since snapshot_get is False
        snap = p.snapshot(update=True)
        self.assertEqual(p._get_count, 0)
        snap_expected = {
            'name': name,
            'label': label,
            'unit': unit,
            'vals': repr(vals.Numbers(5, 10)),
            'metadata': metadata
        }
        for k, v in snap_expected.items():
            self.assertEqual(snap[k], v)

        # attributes only available in MultiParameter
        for attr in ['names', 'labels', 'setpoints', 'setpoint_names',
                     'setpoint_labels', 'full_names']:
            self.assertFalse(hasattr(p, attr), attr)

    def test_number_of_validations(self):

        p = Parameter('p', set_cmd=None, initial_value=0,
                      vals=BookkeepingValidator())

        self.assertEqual(p.vals.values_validated, [0])

        p.step = 1
        p.set(10)

        self.assertEqual(p.vals.values_validated,
                         [0, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_snapshot_value(self):
        p_snapshot = Parameter('no_snapshot', set_cmd=None, get_cmd=None,
                               snapshot_value=True)
        p_snapshot(42)
        snap = p_snapshot.snapshot()
        self.assertIn('value', snap)
        p_no_snapshot = Parameter('no_snapshot', set_cmd=None, get_cmd=None,
                                  snapshot_value=False)
        p_no_snapshot(42)
        snap = p_no_snapshot.snapshot()
        self.assertNotIn('value', snap)

    def test_has_set_get(self):
        # Create parameter that has no set_cmd, and get_cmd returns last value
        gettable_parameter = Parameter('1', set_cmd=False, get_cmd=None)
        self.assertTrue(hasattr(gettable_parameter, 'get'))
        self.assertFalse(hasattr(gettable_parameter, 'set'))
        with self.assertRaises(NotImplementedError):
            gettable_parameter(1)
        # Initial value is None if not explicitly set
        self.assertIsNone(gettable_parameter())

        # Create parameter that saves value during set, and has no get_cmd
        settable_parameter = Parameter('2', set_cmd=None, get_cmd=False)
        self.assertFalse(hasattr(settable_parameter, 'get'))
        self.assertTrue(hasattr(settable_parameter, 'set'))
        with self.assertRaises(NotImplementedError):
            settable_parameter()
        settable_parameter(42)

        settable_gettable_parameter = Parameter('3', set_cmd=None, get_cmd=None)
        self.assertTrue(hasattr(settable_gettable_parameter, 'set'))
        self.assertTrue(hasattr(settable_gettable_parameter, 'get'))
        self.assertIsNone(settable_gettable_parameter())
        settable_gettable_parameter(22)
        self.assertEqual(settable_gettable_parameter(), 22)

    def test_str_representation(self):
        # three cases where only name gets used for full_name
        for instrument in blank_instruments:
            p = Parameter(name='fred')
            p._instrument = instrument
            self.assertEqual(str(p), 'fred')

        # and finally an instrument that really has a name
        p = Parameter(name='wilma')
        p._instrument = named_instrument
        self.assertEqual(str(p), 'astro_wilma')

    def test_bad_validator(self):
        with self.assertRaises(TypeError):
            Parameter('p', vals=[1, 2, 3])

    def test_step_ramp(self):
        p = MemoryParameter(name='test_step')
        p(42)
        self.assertListEqual(p.set_values, [42])
        p.step = 1

        self.assertListEqual(p.get_ramp_values(44.5, 1), [43, 44, 44.5])

        p(44.5)
        self.assertListEqual(p.set_values, [42, 43, 44, 44.5])

    def test_scale_raw_value(self):
        p = Parameter(name='test_scale_raw_value', set_cmd=None)
        p(42)
        self.assertEqual(p.raw_value, 42)

        p.scale = 2
        self.assertEqual(p.raw_value, 42) # No set/get cmd performed
        self.assertEqual(p(), 21)

        p(10)
        self.assertEqual(p.raw_value, 20)
        self.assertEqual(p(), 10)

    def test_latest_value(self):
        p = MemoryParameter(name='test_latest_value', get_cmd=lambda: 21)

        p(42)
        self.assertEqual(p.get_latest(), 42)
        self.assertListEqual(p.get_values, [])

        p.get_latest.max_val_age = 0.1
        sleep(0.2)
        self.assertEqual(p.get_latest(), 21)
        self.assertEqual(p.get_values, [21])

    @given(scale=hst.integers(1, 100),
           value=hst.floats(min_value=1e-9, max_value=10))
    def test_ramp_scaled(self, scale, value):
        start_point = 0.0
        p = MemoryParameter(name='p', scale=scale,
                      initial_value=start_point)
        assert p() == start_point
        # first set a step size
        p.step = 0.1
        # and a wait time
        p.inter_delay = 1e-9 # in seconds
        first_step = 1.0
        second_step = 10.0
        # do a step to start from a non zero starting point where
        # scale matters
        p.set(first_step)
        np.testing.assert_allclose(np.array([p.get()]),
                                   np.array([first_step]))

        expected_raw_steps = np.linspace(start_point*scale, first_step*scale, 11)
        # getting the raw values that are actually send to the instrument.
        # these are scaled in the set_wrapper
        np.testing.assert_allclose(np.array(p.set_values), expected_raw_steps)
        assert p.raw_value == first_step*scale
        # then check the generated steps. They should not be scaled as the
        # scaling happens when setting them
        expected_steps = np.linspace(first_step+p.step,
                                     second_step,90)
        np.testing.assert_allclose(p.get_ramp_values(second_step, p.step),
                                   expected_steps)
        p.set(10)
        np.testing.assert_allclose(np.array(p.set_values),
                                   np.linspace(0.0*scale, 10*scale, 101))
        p.set(value)
        np.testing.assert_allclose(p.get(), value)
        assert p.raw_value == value * scale

    @given(value=hst.floats(min_value=1e-9, max_value=10))
    def test_ramp_parser(self, value):
        start_point = 0.0
        p = MemoryParameter(name='p',
                            set_parser=lambda x: -x,
                            get_parser=lambda x: -x,
                            initial_value=start_point)
        assert p() == start_point
        # first set a step size
        p.step = 0.1
        # and a wait time
        p.inter_delay = 1e-9 # in seconds
        first_step = 1.0
        second_step = 10.0
        # do a step to start from a non zero starting point where
        # scale matters
        p.set(first_step)
        assert p.get() == first_step
        assert p.raw_value == - first_step
        np.testing.assert_allclose(np.array([p.get()]),
                                   np.array([first_step]))

        expected_raw_steps = np.linspace(-start_point, -first_step, 11)
        # getting the raw values that are actually send to the instrument.
        # these are parsed in the set_wrapper
        np.testing.assert_allclose(np.array(p.set_values), expected_raw_steps)
        assert p.raw_value == -first_step
        # then check the generated steps. They should not be parsed as the
        # scaling happens when setting them
        expected_steps = np.linspace((first_step+p.step),
                                     second_step,90)
        np.testing.assert_allclose(p.get_ramp_values(second_step, p.step),
                                   expected_steps)
        p.set(second_step)
        np.testing.assert_allclose(np.array(p.set_values),
                                   np.linspace(-start_point, -second_step, 101))
        p.set(value)
        np.testing.assert_allclose(p.get(), value)
        assert p.raw_value == - value



    @given(scale=hst.integers(1, 100),
           value=hst.floats(min_value=1e-9, max_value=10))
    def test_ramp_parsed_scaled(self, scale, value):
        start_point = 0.0
        p = MemoryParameter(name='p',
                            scale = scale,
                            set_parser=lambda x: -x,
                            get_parser=lambda x: -x,
                            initial_value=start_point)
        assert p() == start_point
        # first set a step size
        p.step = 0.1
        # and a wait time
        p.inter_delay = 1e-9 # in seconds
        first_step = 1.0
        second_step = 10.0
        p.set(first_step)
        assert p.get() == first_step
        assert p.raw_value == - first_step * scale
        expected_raw_steps = np.linspace(-start_point*scale, -first_step*scale, 11)
        # getting the raw values that are actually send to the instrument.
        # these are parsed in the set_wrapper
        np.testing.assert_allclose(np.array(p.set_values), expected_raw_steps)
        assert p.raw_value == - scale * first_step
        expected_steps = np.linspace(first_step+p.step,second_step,90)
        np.testing.assert_allclose(p.get_ramp_values(10, p.step),
                                   expected_steps)
        p.set(second_step)
        np.testing.assert_allclose(np.array(p.set_values),
                                   np.linspace(-start_point*scale, -second_step*scale, 101))
        p.set(value)
        np.testing.assert_allclose(p.get(), value)
        assert p.raw_value == -scale * value


class TestValsandParseParameter(TestCase):

    def setUp(self):
        self.parameter = Parameter(name='foobar',
                                   set_cmd=None, get_cmd=None,
                                   set_parser=lambda x: int(round(x)),
                                   vals=vals.PermissiveInts(0))

    def test_setting_int_with_float(self):

        a = 0
        b = 10
        values = np.linspace(a, b, b-a+1)
        for i in values:
            self.parameter(i)
            a = self.parameter()
            assert isinstance(a, int)

    def test_setting_int_with_float_not_close(self):

        a = 0
        b = 10
        values = np.linspace(a, b, b-a+2)
        for i in values[1:-2]:
            with self.assertRaises(TypeError):
                self.parameter(i)


class SimpleArrayParam(ArrayParameter):
    def __init__(self, return_val, *args, **kwargs):
        self._return_val = return_val
        self._get_count = 0
        super().__init__(*args, **kwargs)

    def get_raw(self):
        self._get_count += 1
        self._save_val(self._return_val)
        return self._return_val


class SettableArray(SimpleArrayParam):
    # this is not allowed - just created to raise an error in the test below
    def set_raw(self, v):
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

        self.assertEqual(str(p), name)

        self.assertEqual(p._get_count, 0)
        snap = p.snapshot(update=True)
        self.assertEqual(p._get_count, 0)
        snap_expected = {
            'name': name,
            'label': name
        }
        for k, v in snap_expected.items():
            self.assertEqual(snap[k], v)

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
            'value': [6, 7]
        }
        for k, v in snap_expected.items():
            self.assertEqual(snap[k], v)

        self.assertIn(name, p.__doc__)
        self.assertIn(docstring, p.__doc__)

    def test_has_set_get(self):
        name = 'array_param'
        shape = (3,)
        with self.assertRaises(AttributeError):
            ArrayParameter(name, shape)

        p = SimpleArrayParam([1, 2, 3], name, shape)

        self.assertTrue(hasattr(p, 'get'))
        self.assertFalse(hasattr(p, 'set'))

        with self.assertRaises(AttributeError):
            SettableArray([1, 2, 3], name, shape)

    def test_full_name(self):
        # three cases where only name gets used for full_name
        for instrument in blank_instruments:
            p = SimpleArrayParam([6, 7], 'fred', (2,))
            p._instrument = instrument
            self.assertEqual(str(p), 'fred')

        # and finally an instrument that really has a name
        p = SimpleArrayParam([6, 7], 'wilma', (2,))
        p._instrument = named_instrument
        self.assertEqual(str(p), 'astro_wilma')

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

    def get_raw(self):
        self._get_count += 1
        self._save_val(self._return_val)
        return self._return_val


class SettableMulti(SimpleMultiParam):
    # this is not fully suported - just created to raise a warning in the test below.
    # We test that the warning is raised
    def set_raw(self, v):
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

        self.assertEqual(str(p), name)

        self.assertEqual(p._get_count, 0)
        snap = p.snapshot(update=True)
        self.assertEqual(p._get_count, 0)
        snap_expected = {
            'name': name,
            'names': names,
            'labels': names,
            'units': [''] * 3
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
                             docstring=docstring, snapshot_value=True,
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
        self.assertEqual(p._get_count, 1)
        snap_expected = {
            'name': name,
            'names': names,
            'labels': labels,
            'units': units,
            'setpoint_names': setpoint_names,
            'setpoint_labels': setpoint_labels,
            'metadata': metadata,
            'value': [0, [1, 2, 3], [[4, 5], [6, 7]]]
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

        self.assertTrue(hasattr(p, 'get'))
        self.assertFalse(hasattr(p, 'set'))
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
            self.assertEqual(str(p), name)

            self.assertEqual(p.full_names, names)

        # and finally an instrument that really has a name
        p = SimpleMultiParam([0, [1, 2, 3], [[4, 5], [6, 7]]],
                             name, names, shapes)
        p._instrument = named_instrument
        self.assertEqual(str(p), 'astro_mixed_dimensions')

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
        p = Parameter('test', get_cmd=None, set_cmd=None)

        def doubler(x):
            p.set(x * 2)

        f = Function('f', call_cmd=doubler, args=[vals.Numbers(-10, 10)])

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
        p = Parameter('p_int', get_cmd=self.get_p, get_parser=int,
                              set_cmd=self.set_p, set_parser=self.parse_set_p)

        p(5)
        self.assertEqual(self._p, '5')
        self.assertEqual(p(), 5)

    def test_settable(self):
        p = Parameter('p', set_cmd=self.set_p, get_cmd=False)

        p(10)
        self.assertEqual(self._p, 10)
        with self.assertRaises(NotImplementedError):
            p()

        self.assertTrue(hasattr(p, 'set'))
        self.assertFalse(hasattr(p, 'get'))

    def test_gettable(self):
        p = Parameter('p', get_cmd=self.get_p)
        self._p = 21

        self.assertEqual(p(), 21)
        self.assertEqual(p.get(), 21)

        with self.assertRaises(NotImplementedError):
            p(10)

        self.assertTrue(hasattr(p, 'get'))
        self.assertFalse(hasattr(p, 'set'))

    def test_val_mapping_basic(self):
        p = Parameter('p', set_cmd=self.set_p, get_cmd=self.get_p,
                      val_mapping={'off': 0, 'on': 1},
                      vals=vals.Enum('off', 'on'))

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
        # set_parser with val_mapping
        Parameter('p', set_cmd=self.set_p, get_cmd=self.get_p,
                  val_mapping={'off': 0, 'on': 1},
                  set_parser=self.parse_set_p)

        # get_parser with val_mapping
        p = Parameter('p', set_cmd=self.set_p_prefixed,
                      get_cmd=self.get_p, get_parser=self.strip_prefix,
                      val_mapping={'off': 0, 'on': 1},
                      vals=vals.Enum('off', 'on'))

        p('off')
        self.assertEqual(self._p, 'PVAL: 0')
        self.assertEqual(p(), 'off')

        self._p = 'PVAL: 1'
        self.assertEqual(p(), 'on')


class TestManualParameterValMapping(TestCase):
    def setUp(self):
        self.instrument = DummyInstrument('dummy_holder')

    def tearDown(self):
        self.instrument.close()
        del self.instrument


    def test_val_mapping(self):
        self.instrument.add_parameter('myparameter', set_cmd=None, get_cmd=None, val_mapping={'A': 0, 'B': 1})
        self.instrument.myparameter('A')
        assert self.instrument.myparameter() == 'A'
        assert self.instrument.myparameter() == 'A'
        assert self.instrument.myparameter.raw_value == 0


class TestInstrumentRefParameter(TestCase):

    def setUp(self):
        self.a = DummyInstrument('dummy_holder')
        self.d = DummyInstrument('dummy')

    def test_get_instr(self):
        self.a.add_parameter('test', parameter_class=InstrumentRefParameter)

        self.a.test.set(self.d.name)

        self.assertEqual(self.a.test.get(), self.d.name)
        self.assertEqual(self.a.test.get_instr(), self.d)

    def tearDown(self):
        self.a.close()
        self.d.close()
        del self.a
        del self.d


class TestCopyParameter(TestCase):
    def test_copy_parameter(self):
        p1 = Parameter(name='p1', initial_value=42, set_cmd=None)
        p2 = copy(p1)

        self.assertEqual(p1.raw_value, 42)
        self.assertEqual(p1(), 42)
        self.assertEqual(p2.raw_value, 42)
        self.assertEqual(p2(), 42)

        p1(43)
        self.assertEqual(p1.raw_value, 43)
        self.assertEqual(p1(), 43)
        self.assertEqual(p2.raw_value, 42)
        self.assertEqual(p2(), 42)

        p2(44)
        self.assertEqual(p1.raw_value, 43)
        self.assertEqual(p1(), 43)
        self.assertEqual(p2.raw_value, 44)
        self.assertEqual(p2(), 44)

    def test_deepcopy_parameter(self):
        p1 = Parameter(name='p1', initial_value=42, set_cmd=None)
        p2 = deepcopy(p1)

        self.assertEqual(p1.raw_value, 42)
        self.assertEqual(p1(), 42)
        self.assertEqual(p2.raw_value, 42)
        self.assertEqual(p2(), 42)

        p1(43)
        self.assertEqual(p1.raw_value, 43)
        self.assertEqual(p1(), 43)
        self.assertEqual(p2.raw_value, 42)
        self.assertEqual(p2(), 42)

        p2(44)
        self.assertEqual(p1.raw_value, 43)
        self.assertEqual(p1(), 43)
        self.assertEqual(p2.raw_value, 44)
        self.assertEqual(p2(), 44)

    def test_copy_parameter_change_name(self):
        p = Parameter(name='parameter1')
        p_copy = copy(p)

        self.assertEqual(p_copy.name, 'parameter1')

        p.name = 'changed_parameter_name'
        self.assertEqual(p.name, 'changed_parameter_name')
        self.assertEqual(p_copy.name, 'parameter1')

        p_copy.name = 'new_name'
        self.assertEqual(p.name, 'changed_parameter_name')
        self.assertEqual(p_copy.name, 'new_name')

    def test_deepcopy_parameter_change_name(self):
        p = Parameter(name='parameter1')
        p_copy = deepcopy(p)

        self.assertEqual(p_copy.name, 'parameter1')

        p.name = 'changed_parameter_name'
        self.assertEqual(p.name, 'changed_parameter_name')
        self.assertEqual(p_copy.name, 'parameter1')

        p_copy.name = 'new_name'
        self.assertEqual(p.name, 'changed_parameter_name')
        self.assertEqual(p_copy.name, 'new_name')

    def test_parameter_copy_get_latest(self):
        p = Parameter(name='p1', set_cmd=None)
        p(42)
        p_copy = copy(p)

        self.assertEqual(p_copy.get_latest(), 42)

        p(41)
        self.assertEqual(p.get_latest(), 41)
        self.assertEqual(p_copy.get_latest(), 42)

        p_copy(43)
        self.assertEqual(p.get_latest(), 41)
        self.assertEqual(p_copy.get_latest(), 43)

    def test_parameter_deepcopy_get_latest(self):
        p = Parameter(name='p1', set_cmd=None)
        p(42)
        p_copy = deepcopy(p)

        self.assertEqual(p_copy.get_latest(), 42)

        p(41)
        self.assertEqual(p.get_latest(), 41)
        self.assertEqual(p_copy.get_latest(), 42)

        p_copy(43)
        self.assertEqual(p.get_latest(), 41)
        self.assertEqual(p_copy.get_latest(), 43)

    def test_copy_multi_parameter(self):
        class CustomMultiParameter(MultiParameter):
            def __init__(self, values=None, **kwargs):
                self.values = values
                super().__init__(**kwargs)

            def get_raw(self):
                return self.values

            def set_raw(self, values):
                self.values = values

        custom_multi_parameter = CustomMultiParameter(values=[1,2],
                                                      name='custom_multi',
                                                      names=['arg1', 'arg2'],
                                                      shapes=((),())
                                                      )
        self.assertListEqual(custom_multi_parameter(), [1,2])

        copied_custom_multi_parameter = copy(custom_multi_parameter)
        custom_multi_parameter([3,4])
        self.assertListEqual(custom_multi_parameter(), [3, 4])
        self.assertListEqual(copied_custom_multi_parameter.get_latest(), [1, 2])
        self.assertListEqual(copied_custom_multi_parameter(), [1, 2])

    def test_deepcopy_multi_parameter(self):
        class CustomMultiParameter(MultiParameter):
            def __init__(self, values=None, **kwargs):
                self.values = values
                super().__init__(**kwargs)

            def get_raw(self):
                return self.values

            def set_raw(self, values):
                self.values = values

        custom_multi_parameter = CustomMultiParameter(values=[1,2],
                                                      name='custom_multi',
                                                      names=['arg1', 'arg2'],
                                                      shapes=((),())
                                                      )
        self.assertListEqual(custom_multi_parameter(), [1,2])

        copied_custom_multi_parameter = deepcopy(custom_multi_parameter)
        custom_multi_parameter([3,4])
        self.assertListEqual(custom_multi_parameter(), [3, 4])
        self.assertListEqual(copied_custom_multi_parameter.get_latest(), [1, 2])
        self.assertListEqual(copied_custom_multi_parameter(), [1, 2])

    def test_copy_array_parameter(self):
        class CustomArrayParameter(ArrayParameter):
            def __init__(self, values=None, **kwargs):
                self.values = values
                super().__init__(**kwargs)

            def get_raw(self):
                return self.values

        custom_array_parameter = CustomArrayParameter(values=[1,2],
                                                      name='custom_multi',
                                                      shape=(2,)
                                                      )
        self.assertListEqual(custom_array_parameter(), [1,2])

        copied_custom_multi_parameter = copy(custom_array_parameter)

    def test_deepcopy_array_parameter(self):
        class CustomArrayParameter(ArrayParameter):
            def __init__(self, values=None, **kwargs):
                self.values = values
                super().__init__(**kwargs)

            def get_raw(self):
                return self.values

        custom_array_parameter = CustomArrayParameter(values=[1,2],
                                                      name='custom_multi',
                                                      shape=(2,)
                                                      )
        self.assertListEqual(custom_array_parameter(), [1,2])

        copied_custom_multi_parameter = deepcopy(custom_array_parameter)

    def test_copy_stateful_parameter(self):
        p = Parameter(set_cmd=None)
        p([])

        p_copy = copy(p)
        self.assertEqual(p(), [])
        self.assertEqual(p_copy(), [])

        p().append(1)
        self.assertEqual(p(), [1])
        self.assertEqual(p_copy(), [])

        p_copy().append(2)
        self.assertEqual(p(), [1])
        self.assertEqual(p_copy(), [2])

    def test_deepcopy_stateful_parameter(self):
        p = Parameter(set_cmd=None)
        p([])

        p_copy = deepcopy(p)
        self.assertEqual(p(), [])
        self.assertEqual(p_copy(), [])

        p().append(1)
        self.assertEqual(p(), [1])
        self.assertEqual(p_copy(), [])

        p_copy().append(2)
        self.assertEqual(p(), [1])
        self.assertEqual(p_copy(), [2])


class TestParameterSignal(TestCase):
    def save_args_kwargs(self, *args, **kwargs):
        self.args_kwargs_dict['args'] = args
        self.args_kwargs_dict['kwargs'] = kwargs

    def setUp(self):
        self.args_kwargs_dict = {}

        self.source_parameter = Parameter(name='source', set_cmd=None,
                                          initial_value=42)

        self.target_parameter = Parameter(name='target', set_cmd=None,
                                          initial_value=43)

    def test_parameter_connect_function(self):
        self.source_parameter.connect(self.save_args_kwargs)

        self.source_parameter(41)
        self.assertEqual(self.args_kwargs_dict['args'], (41,))
        self.assertEqual(self.args_kwargs_dict['kwargs'], {})

    def test_parameter_connect_parameter(self):
        self.source_parameter.connect(self.target_parameter, update=False)
        self.assertEqual(self.target_parameter(), 43)

        self.source_parameter(41)
        self.assertEqual(self.target_parameter(), 41)

        self.target_parameter(43)
        self.assertEqual(self.target_parameter(), 43)

    def test_delete_parameter(self):
        target_ref = weakref.ref(self.target_parameter)

        del self.target_parameter
        gc.collect()
        self.assertIsNone(target_ref())

    def test_delete_connected_parameter(self):
        self.source_parameter.connect(self.target_parameter)

        target_ref = weakref.ref(self.target_parameter)

        del self.target_parameter
        gc.collect()
        self.assertIsNone(target_ref())

    def test_delete_connected_parameter_set(self):
        self.source_parameter.connect(self.target_parameter)
        self.source_parameter(41)
        self.assertEqual(self.target_parameter(), 41)

        target_ref = weakref.ref(self.target_parameter)
        self.assertEqual(len(self.source_parameter.signal.receivers), 1)

        del self.target_parameter
        gc.collect()
        self.assertIsNone(target_ref())
        self.assertEqual(len(self.source_parameter.signal.receivers), 0)

    def test_copied_source_parameter(self):
        self.source_parameter.connect(self.target_parameter, update=False)
        deepcopied_source_parameter = copy(self.source_parameter)

        self.assertEqual(self.target_parameter(), 43)
        deepcopied_source_parameter(41)
        self.assertEqual(self.source_parameter(), 42)
        self.assertEqual(self.target_parameter(), 43)

        self.source_parameter(44)
        self.assertEqual(self.target_parameter(), 44)
        self.assertEqual(deepcopied_source_parameter(), 41)

    def test_deepcopied_source_parameter(self):
        self.source_parameter.connect(self.target_parameter, update=False)
        deepcopied_source_parameter = deepcopy(self.source_parameter)

        self.assertEqual(self.target_parameter(), 43)
        deepcopied_source_parameter(41)
        self.assertEqual(self.source_parameter(), 42)
        self.assertEqual(self.target_parameter(), 43)

        self.source_parameter(44)
        self.assertEqual(self.target_parameter(), 44)
        self.assertEqual(deepcopied_source_parameter(), 41)

    def test_copied_target_parameter(self):
        self.source_parameter.connect(self.target_parameter, update=False)
        copied_target_parameter = copy(self.target_parameter)

        self.assertEqual(self.target_parameter(), 43)
        self.assertEqual(copied_target_parameter(), 43)

        self.source_parameter(41)
        self.assertEqual(self.source_parameter(), 41)
        self.assertEqual(self.target_parameter(), 41)
        self.assertEqual(copied_target_parameter(), 43)

        self.target_parameter(44)
        self.assertEqual(self.source_parameter(), 41)
        self.assertEqual(self.target_parameter(), 44)
        self.assertEqual(copied_target_parameter(), 43)

    def test_deepcopied_target_parameter(self):
        self.source_parameter.connect(self.target_parameter, update=False)
        copied_target_parameter = copy(self.target_parameter)

        self.assertEqual(self.target_parameter(), 43)
        self.assertEqual(copied_target_parameter(), 43)

        self.source_parameter(41)
        self.assertEqual(self.source_parameter(), 41)
        self.assertEqual(self.target_parameter(), 41)
        self.assertEqual(copied_target_parameter(), 43)

        self.target_parameter(44)
        self.assertEqual(self.source_parameter(), 41)
        self.assertEqual(self.target_parameter(), 44)
        self.assertEqual(copied_target_parameter(), 43)

    def test_connected_parameter(self):
        self.source_parameter.connect(self.target_parameter)
        copied_target_parameter = copy(self.target_parameter)

    def test_circular_signalling(self):
        self.set_calls = 0
        def prevent_circular_signalling(val):
            if self.set_calls > 10:
                raise RecursionError('Too many set calls')
            self.set_calls += 1

        self.source_parameter = Parameter(name='source',
                                          initial_value=42,
                                          set_cmd=prevent_circular_signalling)

        self.target_parameter = Parameter(name='target',
                                          initial_value=43,
                                          set_cmd=prevent_circular_signalling)

        self.source_parameter.connect(self.target_parameter)
        self.source_parameter.signal.send(0)

        self.target_parameter.connect(self.source_parameter)

        self.set_calls = 0

        self.source_parameter(40)
        self.assertEqual(self.set_calls, 2)
        self.assertEqual(self.source_parameter(), 40)
        self.assertEqual(self.target_parameter(), 40)

        self.source_parameter(41)
        self.assertEqual(self.set_calls, 4)
        self.assertEqual(self.source_parameter(), 41)
        self.assertEqual(self.target_parameter(), 41)

    def test_connect_to_second_parameter(self):
        self.source_parameter2 = Parameter('source2', initial_value=12,
                                           set_cmd=None)

        self.source_parameter.connect(self.target_parameter)
        self.source_parameter2.connect(self.target_parameter)

        self.source_parameter(1)
        self.assertEqual(self.source_parameter(), 1)
        self.assertEqual(self.source_parameter2(), 12)
        self.assertEqual(self.target_parameter(), 1)

        self.source_parameter2(2)
        self.assertEqual(self.source_parameter(), 1)
        self.assertEqual(self.source_parameter2(), 2)
        self.assertEqual(self.target_parameter(), 2)

        self.source_parameter2.disconnect(self.target_parameter)
        self.source_parameter(3)
        self.source_parameter2(4)
        self.assertEqual(self.source_parameter(), 3)
        self.assertEqual(self.source_parameter2(), 4)
        self.assertEqual(self.target_parameter(), 3)

    def test_disconnect_parameter(self):
        self.source_parameter.connect(self.target_parameter)

        self.source_parameter(123)
        self.assertEqual(self.source_parameter(), 123)
        self.assertEqual(self.target_parameter(), 123)

        self.source_parameter.disconnect(self.target_parameter)

        self.source_parameter(1)
        self.assertEqual(self.source_parameter(), 1)
        self.assertEqual(self.target_parameter(), 123)

    def test_triple_connected_parameters(self):
        self.second_target_parameter = Parameter('p3', initial_value=40,
                                                 set_cmd=None)
        self.source_parameter.connect(self.target_parameter)
        self.target_parameter.connect(self.second_target_parameter)

        self.source_parameter(123)
        self.assertEqual(self.source_parameter(), 123)
        self.assertEqual(self.target_parameter(), 123)
        self.assertEqual(self.second_target_parameter(), 123)

        self.source_parameter.disconnect(self.target_parameter)

        self.source_parameter(1)
        self.assertEqual(self.source_parameter(), 1)
        self.assertEqual(self.target_parameter(), 123)
        self.assertEqual(self.second_target_parameter(), 123)

    def test_config_link(self):
        # Note that config links only work in silq, since it relies on the
        # SubConfig, and so we only test here that it doesn't raise an error
        config_parameter = Parameter('config', config_link='pulses.duration')

    def test_parameter_signal_modifiers(self):
        self.source_parameter.connect(self.target_parameter, offset=1)
        self.source_parameter(10)
        self.assertEqual(self.source_parameter(), 10)
        self.assertEqual(self.target_parameter(), 11)

        self.source_parameter.disconnect(self.target_parameter)
        self.source_parameter(12)
        self.assertEqual(self.source_parameter(), 12)
        self.assertEqual(self.target_parameter(), 11)

        self.source_parameter.connect(self.target_parameter, scale=2)
        self.source_parameter(13)
        self.assertEqual(self.source_parameter(), 13)
        self.assertEqual(self.target_parameter(), 26)

        self.source_parameter.disconnect(self.target_parameter)
        self.source_parameter.connect(self.target_parameter, scale=2,
                                      offset=10)
        self.source_parameter(14)
        self.assertEqual(self.source_parameter(), 14)
        self.assertEqual(self.target_parameter(), 38)

    def test_copied_parameter_signal(self):
        values = []
        def fun(value):
            values.append(value)

        p = Parameter(set_cmd=None)
        p_copy = copy(p)

        self.assertIsNotNone(p_copy.signal)
        p_copy.connect(fun, update=False)
        p_copy(42)
        self.assertListEqual(values, [42])

    def test_deepcopied_parameter_signal(self):
        values = []
        def fun(value):
            values.append(value)

        p = Parameter(set_cmd=None)
        p_copy = deepcopy(p)

        self.assertIsNotNone(p_copy.signal)
        p_copy.connect(fun, update=False)
        p_copy(42)
        self.assertListEqual(values, [42])


class ListHandler(logging.Handler):  # Inherit from logging.Handler
    def __init__(self, log_list):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        # Our custom argument
        self.log_list = log_list

    def emit(self, record):
        # record.message is the log message
        self.log_list.append(record.msg)


class TestParameterLogging(TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.log_list = []
        self.handler = ListHandler(self.log_list)
        logging.getLogger().addHandler(self.handler)
        print('Started logging')

    def tearDown(self):
        logging.getLogger().removeHandler(self.handler)
        print('Stopped logging')

    def test_logging(self):
        p = Parameter('p', initial_value=41, set_cmd=None)
        self.assertEqual(len(self.log_list), 1)
        p(42)
        self.assertEqual(len(self.log_list), 2)

        p.log_changes = False
        p(43)
        self.assertEqual(len(self.log_list), 2)

        p.log_changes = True
        p(44)
        self.assertEqual(len(self.log_list), 3)

        # Set to same value, no log should be emitted
        p(44)
        self.assertEqual(len(self.log_list), 3)


class TestParameterSnapshotting(TestCase):
    def test_empty_parameter_snapshot(self):
        param = Parameter()
        snapshot = param.snapshot()
        self.assertEqual(snapshot['name'], 'None')
        self.assertEqual(snapshot['value'], None)
        self.assertEqual(snapshot['raw_value'], None)

    def test_empty_parameter_simplified_snapshot(self):
        param = Parameter()
        snapshot = param.snapshot(simplify=True)
        self.assertEqual(snapshot, {'name': 'None', 'value': None})

    def test_named_parameter_simplified_snapshot(self):
        param = Parameter('param_1')
        snapshot = param.snapshot(simplify=True)
        self.assertEqual(snapshot, {'name': 'param_1',
                                    'label': 'Param 1',
                                    'value': None})
        param.unit = 'V'
        snapshot = param.snapshot(simplify=True)
        self.assertEqual(snapshot, {'name': 'param_1',
                                    'label': 'Param 1',
                                    'unit': 'V',
                                    'value': None})


class TestParameterPickling(TestCase):
    def test_pickle_empty_parameter(self):
        p = Parameter(name='param1')
        pickle_dump = pickle.dumps(p)
        p_pickled = pickle._loads(pickle_dump)
        self.assertEqual(p_pickled.name, 'param1')

    def test_pickle_empty_parameter_with_set(self):
        p = Parameter(name='param1', set_cmd=lambda x: 'bla')
        pickle_dump = pickle.dumps(p)
        p_pickled = pickle._loads(pickle_dump)
        self.assertEqual(p.name, 'param1')
        self.assertEqual(p_pickled.name, 'param1')

    def test_pickle_empty_parameter_with_get(self):
        p = Parameter(name='param1', get_cmd=lambda: 123)
        p.get()  # Set value to 123
        pickle_dump = pickle.dumps(p)

        p_pickled = pickle._loads(pickle_dump)
        self.assertEqual(p_pickled.name, 'param1')
        self.assertEqual(p_pickled.get_latest(), 123)
        self.assertEqual(p_pickled(), 123)

    def test_pickle_get_latest(self):
        p = Parameter('p1', set_cmd=None)
        p(42)

        pickle_dump = pickle.dumps(p)
        p_pickled = pickle.loads(pickle_dump)

        self.assertEqual(p.get_latest(), 42)
        self.assertEqual(p_pickled.get_latest(), 42)

        p(41)
        self.assertEqual(p.get_latest(), 41)
        self.assertEqual(p_pickled.get_latest(), 42)

        p_pickled._latest['value'] = 43
        self.assertEqual(p.get_latest(), 41)
        self.assertEqual(p_pickled.get_latest(), 43)

    def test_pickle_connected_parameter(self):
        values = []
        def fun(value):
            values.append(value)

        p = Parameter(set_cmd=None)
        p.connect(fun, update=False)

        p(42)
        self.assertListEqual(values, [42])

        pickle_dump = pickle.dumps(p)
        pickled_parameter = pickle.loads(pickle_dump)

        self.assertListEqual(values, [42])
        p(41)
        self.assertListEqual(values, [42, 41])
        self.assertEqual(pickled_parameter(), 42)