"""
Test suite for parameter
"""
from collections import namedtuple, Iterable
from unittest import TestCase
from typing import Tuple
import pytest

import numpy as np
from hypothesis import given, event, settings
import hypothesis.strategies as hst
from qcodes import Function
from qcodes.instrument.parameter import (
    Parameter, ArrayParameter, MultiParameter,
    InstrumentRefParameter)
import qcodes.utils.validators as vals
from qcodes.tests.instrument_mocks import DummyInstrument
from qcodes.utils.validators import Numbers


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
) # type: Tuple
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
    def test_no_name(self):
        with self.assertRaises(TypeError):
            Parameter()

    def test_default_attributes(self):
        # Test the default attributes, providing only a name
        name = 'repetitions'
        p = GettableParam(name, vals=vals.Numbers())
        self.assertEqual(p.name, name)
        self.assertEqual(p.label, name)
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
            'label': name,
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
        # in the set wrapper the final value is validated
        # and then subsequently each step is validated.
        # in this case there is one step so the final value
        # is validated twice.
        self.assertEqual(p.vals.values_validated, [0, 0])

        p.step = 1
        p.set(10)
        self.assertEqual(p.vals.values_validated,
                         [0, 0, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

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

    # There are a number different scenarios for testing a parameter with scale
    # and offset. Therefore a custom strategy for generating test parameters
    # is implemented here. The possible cases are:
    # for getting and setting a parameter: values can be
    #    scalar:
    #        offset and scale can be scalars
    # for getting only:
    #    array:
    #        offset and scale can be scalars or arrays(of same legnth as values)
    #        independently

    # define shorthands for strategies
    TestFloats = hst.floats(min_value=-1e40, max_value=1e40).filter(lambda x: x != 0)
    SharedSize = hst.shared(hst.integers(min_value=1, max_value=100), key='shared_size')
    ValuesScalar = hst.shared(hst.booleans(), key='values_scalar')

    # the following test stra
    @hst.composite
    def iterable_or_number(draw, values, size, values_scalar, is_values):
        if draw(values_scalar):
            # if parameter values are scalar, return scalar for values and scale/offset
            return draw(values)
        elif is_values:
            # if parameter values are not scalar and parameter values are requested
            # return a list of values of the given size
            return draw(hst.lists(values, min_size=draw(size), max_size=draw(size)))
        else:
            # if parameter values are not scalar and scale/offset are requested
            # make a random choice whether to return a list of the same size as the values
            # or a simple scalar
            if draw(hst.booleans()):
                return draw(hst.lists(values, min_size=draw(size), max_size=draw(size)))
            else:
                return draw(values)

    @settings(max_examples=500)  # default:100 increased
    @given(values=iterable_or_number(TestFloats, SharedSize, ValuesScalar, True),
           offsets=iterable_or_number(TestFloats, SharedSize, ValuesScalar, False),
           scales=iterable_or_number(TestFloats, SharedSize, ValuesScalar, False))
    def test_scale_and_offset_raw_value_iterable(self, values, offsets, scales):
        p = Parameter(name='test_scale_and_offset_raw_value', set_cmd=None)

        # test that scale and offset does not change the default behaviour
        p(values)
        self.assertEqual(p.raw_value, values)

        # test setting scale and offset does not change anything
        p.scale = scales
        p.offset = offsets
        self.assertEqual(p.raw_value, values)


        np_values = np.array(values)
        np_offsets = np.array(offsets)
        np_scales = np.array(scales)
        np_get_values = np.array(p())
        np.testing.assert_allclose(np_get_values, (np_values-np_offsets)/np_scales) # No set/get cmd performed

        # test set, only for scalar values
        if not isinstance(values, Iterable):
            p(values)
            np.testing.assert_allclose(np.array(p.raw_value), np_values*np_scales + np_offsets) # No set/get cmd performed

            # testing conversion back and forth
            p(values)
            np_get_values = np.array(p())
            np.testing.assert_allclose(np_get_values, np_values) # No set/get cmd performed

        # adding statistics
        if isinstance(offsets, Iterable):
            event('Offset is array')
        if isinstance(scales, Iterable):
            event('Scale is array')
        if isinstance(values, Iterable):
            event('Value is array')
        if isinstance(scales, Iterable) and isinstance(offsets, Iterable):
            event('Scale is array and also offset')
        if isinstance(scales, Iterable) and not isinstance(offsets, Iterable):
            event('Scale is array but not offset')


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

    def test_steppeing_from_invalid_starting_point(self):

        the_value = -10

        def set_function(value):
            nonlocal the_value
            the_value = value

        def get_function():
            return the_value

        a = Parameter('test', set_cmd=set_function, get_cmd=get_function,
                      vals=Numbers(0, 100), step=5)
        # We start out by setting the parameter to an
        # invalid value. This is not possible using initial_value
        # as the validator will catch that but perhaps this may happen
        # if the instrument can return out of range values.
        assert a.get() == -10
        with pytest.raises(ValueError):
            # trying to set to 10 should raise even with 10 valid
            # as the steps demand that we first step to -5 which is not
            a.set(10)
        # afterwards the value should still be the same
        assert a.get() == -10


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
            'label': name,
            'unit': ''
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


class TestSetContextManager(TestCase):

    def setUp(self):
        self.instrument = DummyInstrument('dummy_holder')
        self.instrument.add_parameter(
            "a",
            set_cmd=None,
            get_cmd=None
        )

    def tearDown(self):
        self.instrument.close()
        del self.instrument

    def test_none_value(self):
        with self.instrument.a.set_to(3):
            assert self.instrument.a.get() == 3
        assert self.instrument.a.get() is None

    def test_context(self):
        self.instrument.a.set(2)

        with self.instrument.a.set_to(3):
            assert self.instrument.a.get() == 3
        assert self.instrument.a.get() == 2
