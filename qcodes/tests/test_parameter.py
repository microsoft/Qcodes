"""
Test suite for parameter
"""
from collections import namedtuple
from collections.abc import Iterable
from unittest import TestCase
from typing import Tuple
import pytest
from datetime import datetime, timedelta
import time
from functools import partial

import numpy as np
from hypothesis import given, event, settings
import hypothesis.strategies as hst
from qcodes import Function
from qcodes.instrument.parameter import (
    Parameter, ArrayParameter, MultiParameter, ManualParameter,
    InstrumentRefParameter, ScaledParameter, DelegateParameter,
    _BaseParameter)
import qcodes.utils.validators as vals
from qcodes.tests.instrument_mocks import DummyInstrument
from qcodes.utils.helpers import create_on_off_val_mapping
from qcodes.utils.validators import Numbers


class GettableParam(Parameter):
    """ Parameter that keeps track of number of get operations"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._get_count = 0

    def get_raw(self):
        self._get_count += 1
        return 42

class BetterGettableParam(Parameter):
    """ Parameter that keeps track of number of get operations,
        But can actually store values"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._get_count = 0

    def get_raw(self):
        self._get_count += 1
        return self.cache._raw_value


class DeprecatedParam(Parameter):
    """ Parameter that uses deprecated wrapping of get and set"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._value = 42
        self.set_count = 0
        self.get_count = 0

    def get(self):
        self.get_count += 1
        return self._value

    def set(self, value):
        self.set_count += 1
        self._value = value


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
                val = self.cache._raw_value
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

    def test_number_of_validations_for_set_cache(self):
        p = Parameter('p', set_cmd=None,
                      vals=BookkeepingValidator())
        self.assertEqual(p.vals.values_validated, [])

        p.cache.set(1)
        self.assertEqual(p.vals.values_validated, [1])

        p.cache.set(4)
        self.assertEqual(p.vals.values_validated, [1, 4])

        p.step = 1
        p.cache.set(10)
        self.assertEqual(p.vals.values_validated, [1, 4, 10])

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

    def test_get_latest(self):
        time_resolution = time.get_clock_info('time').resolution
        sleep_delta = 2 * time_resolution

        # Create a gettable parameter
        local_parameter = Parameter('test_param', set_cmd=None, get_cmd=None)
        before_set = datetime.now()
        time.sleep(sleep_delta)
        local_parameter.set(1)
        time.sleep(sleep_delta)
        after_set = datetime.now()

        # Check we return last set value, with the correct timestamp
        self.assertEqual(local_parameter.get_latest(), 1)
        self.assertTrue(before_set < local_parameter.get_latest.get_timestamp() < after_set)

        # Check that updating the value updates the timestamp
        time.sleep(sleep_delta)
        local_parameter.set(2)
        self.assertEqual(local_parameter.get_latest(), 2)
        self.assertGreater(local_parameter.get_latest.get_timestamp(), after_set)

    def test_get_latest_raw_value(self):
        # To have a simple distinction between raw value and value of the
        # parameter lets create a parameter with an offset
        p = Parameter('p', set_cmd=None, get_cmd=None, offset=42)
        assert p.get_latest.get_timestamp() is None

        # Initially, the parameter's raw value is None
        assert p.get_latest.get_raw_value() is None

        # After setting the parameter to some value, the
        # ``.get_latest.get_raw_value()`` call should return the new raw value
        # of the parameter
        p(3)
        assert p.get_latest.get_timestamp() is not None
        assert p.get_latest.get() == 3
        assert p.get_latest() == 3
        assert p.get_latest.get_raw_value() == 3 + 42

    def test_get_latest_unknown(self):
        """
        Test that get latest on a parameter that has not been acquired will
        trigger a get
        """
        value = 1
        local_parameter = BetterGettableParam('test_param', set_cmd=None,
                                              get_cmd=None)
        # fake a parameter that has a value but never been get/set to mock
        # an instrument.
        local_parameter.cache._value = value
        local_parameter.cache._raw_value = value
        assert local_parameter.get_latest.get_timestamp() is None
        before_get = datetime.now()
        assert local_parameter._get_count == 0
        assert local_parameter.get_latest() == value
        assert local_parameter._get_count == 1
        # calling get_latest above will call get since TS is None
        # and the TS will therefore no longer be None
        assert local_parameter.get_latest.get_timestamp() is not None
        assert local_parameter.get_latest.get_timestamp() >= before_get
        # calling get_latest now will not trigger get
        assert local_parameter.get_latest() == value
        assert local_parameter._get_count == 1

    def test_get_latest_known(self):
        """
        Test that get latest on a parameter that has a known value will not
        trigger a get
        """
        value = 1
        local_parameter = BetterGettableParam('test_param', set_cmd=None,
                                              get_cmd=None)
        # fake a parameter that has a value acquired 10 sec ago
        start = datetime.now()
        set_time = start - timedelta(seconds=10)
        local_parameter.cache._update_with(
            value=value, raw_value=value, timestamp=set_time)
        assert local_parameter._get_count == 0
        assert local_parameter.get_latest.get_timestamp() == set_time
        assert local_parameter.get_latest() == value
        # calling get_latest above will not call get since TS is set and
        # max_val_age is not
        assert local_parameter._get_count == 0
        assert local_parameter.get_latest.get_timestamp() == set_time

    def test_get_latest_no_get(self):
        """
        Test that get_latest on a parameter that does not have get is handled
        correctly.
        """
        local_parameter = Parameter('test_param', set_cmd=None, get_cmd=False)
        # The parameter does not have a get method.
        with self.assertRaises(AttributeError):
            local_parameter.get()
        # get_latest will fail as get cannot be called and no cache
        # is available
        with self.assertRaises(RuntimeError):
            local_parameter.get_latest()
        value = 1
        local_parameter.set(value)
        assert local_parameter.get_latest() == value

        local_parameter2 = Parameter('test_param2', set_cmd=None,
                                     get_cmd=False, initial_value=value)
        with self.assertRaises(AttributeError):
            local_parameter2.get()
        assert local_parameter2.get_latest() == value

    def test_max_val_age(self):
        value = 1
        start = datetime.now()
        local_parameter = BetterGettableParam('test_param',
                                              set_cmd=None,
                                              max_val_age=1,
                                              initial_value=value)
        assert local_parameter.cache.max_val_age == 1
        assert local_parameter._get_count == 0
        assert local_parameter.get_latest() == value
        assert local_parameter._get_count == 0
        # now fake the time stamp so get should be triggered
        set_time = start - timedelta(seconds=10)
        local_parameter.cache._update_with(
            value=value, raw_value=value, timestamp=set_time)
        # now that ts < max_val_age calling get_latest should update the time
        assert local_parameter.get_latest.get_timestamp() == set_time
        assert local_parameter.get_latest() == value
        assert local_parameter._get_count == 1
        assert local_parameter.get_latest.get_timestamp() >= start

    def test_no_get_max_val_age(self):
        """
        Test that get_latest on a parameter with max_val_age set and
        no get cmd raises correctly.
        """
        value = 1
        with self.assertRaises(SyntaxError):
            _ = Parameter('test_param', set_cmd=None,
                          get_cmd=False,
                          max_val_age=1, initial_value=value)

        # _BaseParameter does not have this check on creation time since get_cmd could be added
        # in a subclass. Here we create a subclass that does add a get command and alsoo does 
        # not implement the check for max_val_age
        class LocalParameter(_BaseParameter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_raw = lambda x: x
                self.set = self._wrap_set(self.set_raw)

        localparameter = LocalParameter('test_param',
                                        None,
                                        max_val_age=1)
        with self.assertRaises(RuntimeError):
            localparameter.get_latest()

    def test_latest_dictionary_gets_updated_upon_set_of_memory_parameter(self):
        p = Parameter('p', set_cmd=None, get_cmd=None)
        assert p.cache._value is None
        assert p.cache._raw_value is None
        assert p.cache.timestamp is None

        p(42)

        assert p.cache._value == 42
        assert p.cache._raw_value == 42
        assert p.cache.timestamp is not None

    def test_has_set_get(self):
        # Create parameter that has no set_cmd, and get_cmd returns last value
        gettable_parameter = Parameter('one', set_cmd=False, get_cmd=None)
        self.assertTrue(hasattr(gettable_parameter, 'get'))
        self.assertFalse(hasattr(gettable_parameter, 'set'))
        with self.assertRaises(NotImplementedError):
            gettable_parameter(1)
        # Initial value is None if not explicitly set
        self.assertIsNone(gettable_parameter())
        # Assert the ``cache.set`` still works for non-settable parameter
        gettable_parameter.cache.set(1)
        self.assertEqual(gettable_parameter(), 1)

        # Create parameter that saves value during set, and has no get_cmd
        settable_parameter = Parameter('two', set_cmd=None, get_cmd=False)
        self.assertFalse(hasattr(settable_parameter, 'get'))
        self.assertTrue(hasattr(settable_parameter, 'set'))
        with self.assertRaises(NotImplementedError):
            settable_parameter()
        settable_parameter(42)

        settable_gettable_parameter = Parameter('three', set_cmd=None, get_cmd=None)
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

    def test_bad_name(self):
        with self.assertRaises(ValueError):
            Parameter('p with space')
        with self.assertRaises(ValueError):
            Parameter('⛄')
        with self.assertRaises(ValueError):
            Parameter('1')


    def test_step_ramp(self):
        p = MemoryParameter(name='test_step')
        p(42)
        self.assertListEqual(p.set_values, [42])
        p.step = 1

        self.assertListEqual(p.get_ramp_values(44.5, 1), [43, 44, 44.5])

        p(44.5)
        self.assertListEqual(p.set_values, [42, 43, 44, 44.5])

        # Assert that stepping does not impact ``cache.set`` call, and that
        # the value that is passed to ``cache.set`` call does not get
        # propagated to parameter's ``set_cmd``
        p.cache.set(40)
        self.assertEqual(p.get_latest(), 40)
        self.assertListEqual(p.set_values, [42, 43, 44, 44.5])

        # Test error conditions
        with self.assertLogs(level='WARN'):
            self.assertEqual(p.get_ramp_values("A", 1), [])
        with self.assertRaises(RuntimeError):
            p.get_ramp_values((1, 2, 3), 1)

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

    @settings(max_examples=300)
    @given(
        values=iterable_or_number(
            TestFloats, SharedSize, ValuesScalar, True),
        offsets=iterable_or_number(
            TestFloats, SharedSize, ValuesScalar, False),
        scales=iterable_or_number(
            TestFloats, SharedSize, ValuesScalar, False))
    def test_scale_and_offset_raw_value_iterable_for_set_cache(
            self, values, offsets, scales):
        p = Parameter(name='test_scale_and_offset_raw_value', set_cmd=None)

        # test that scale and offset does not change the default behaviour
        p.cache.set(values)
        self.assertEqual(p.raw_value, values)

        # test setting scale and offset does not change anything
        p.scale = scales
        p.offset = offsets
        self.assertEqual(p.raw_value, values)

        np_values = np.array(values)
        np_offsets = np.array(offsets)
        np_scales = np.array(scales)
        np_get_latest_values = np.array(p.get_latest())
        # Without a call to ``get``, ``get_latest`` will just return old
        # cached values without applying the set scale and offset
        np.testing.assert_allclose(np_get_latest_values, np_values)
        np_get_values = np.array(p.get())
        # Now that ``get`` is called, the returned values are the result of
        # application of the scale and offset. Obviously, calling
        # ``get_latest`` now will also return the values with the applied
        # scale and offset
        np.testing.assert_allclose(np_get_values,
                                   (np_values - np_offsets) / np_scales)
        np_get_latest_values_after_get = np.array(p.get_latest())
        np.testing.assert_allclose(np_get_latest_values_after_get,
                                   (np_values - np_offsets) / np_scales)

        # test ``cache.set`` for scalar values
        if not isinstance(values, Iterable):
            p.cache.set(values)
            np.testing.assert_allclose(np.array(p.raw_value),
                                       np_values * np_scales + np_offsets)
            # No set/get cmd performed

            # testing conversion back and forth
            p.cache.set(values)
            np_get_latest_values = np.array(p.get_latest())
            # No set/get cmd performed
            np.testing.assert_allclose(np_get_latest_values, np_values)

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


_P = Parameter


@pytest.mark.parametrize(
    argnames=('p', 'value', 'raw_value'),
    argvalues=(
        (_P('p', set_cmd=None, get_cmd=None), 4, 4),
        (_P('p', set_cmd=False, get_cmd=None), 14, 14),
        (_P('p', set_cmd=None, get_cmd=False), 14, 14),
        (_P('p', set_cmd=None, get_cmd=None, vals=vals.OnOff()), 'on', 'on'),
        (_P('p', set_cmd=None, get_cmd=None, val_mapping={'screw': 1}),
         'screw', 1),
        (_P('p', set_cmd=None, get_cmd=None, set_parser=str, get_parser=int),
         14, '14'),
        (_P('p', set_cmd=None, get_cmd=None, step=7), 14, 14),
        (_P('p', set_cmd=None, get_cmd=None, offset=3), 14, 17),
        (_P('p', set_cmd=None, get_cmd=None, scale=2), 14, 28),
        (_P('p', set_cmd=None, get_cmd=None, offset=-3, scale=2), 14, 25),
    ),
    ids=(
        'with_nothing_extra',
        'without_set_cmd',
        'without_get_cmd',
        'with_on_off_validator',
        'with_val_mapping',
        'with_set_and_parsers',
        'with_step',
        'with_offset',
        'with_scale',
        'with_scale_and_offset',
    )
)
def test_set_latest_works_for_plain_memory_parameter(p, value, raw_value):
    # Set latest value of the parameter
    p.cache.set(value)

    # Assert the latest value and raw_value
    assert p.get_latest() == value
    assert p.raw_value == raw_value

    # Assert latest value and raw_value via private attributes for strictness
    assert p.cache._value == value
    assert p.cache._raw_value == raw_value

    # Now let's get the value of the parameter to ensure that the value that
    # is set above gets picked up from the `_latest` dictionary (due to
    # `get_cmd=None`)

    if not hasattr(p, 'get'):
        return  # finish the test here for non-gettable parameters

    gotten_value = p.get()

    assert gotten_value == value

    # Assert the latest value and raw_value
    assert p.get_latest() == value
    assert p.raw_value == raw_value

    # Assert latest value and raw_value via private attributes for strictness
    assert p.cache._value == value
    assert p.cache._raw_value == raw_value


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


class SimpleMultiParam(MultiParameter):
    def __init__(self, return_val, *args, **kwargs):
        self._return_val = return_val
        self._get_count = 0
        super().__init__(*args, **kwargs)

    def get_raw(self):
        self._get_count += 1
        return self._return_val


class SettableMulti(SimpleMultiParam):
    def set_raw(self, v):
        print("Calling set")
        self._return_val = v


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

        original_value = [0, [1, 2, 3], [[4, 5], [6, 7]]]
        p = SimpleMultiParam(original_value, name, names, shapes)

        self.assertTrue(hasattr(p, 'get'))
        self.assertFalse(hasattr(p, 'set'))
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
        self.assertTrue(hasattr(p, 'set'))
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

        p.cache.set(7)
        self.assertEqual(p.get_latest(), 7)
        # Nothing has been passed to the "instrument" at ``cache.set``
        # call, hence the following assertions should hold
        self.assertEqual(self._p, '5')
        self.assertEqual(p(), 5)
        self.assertEqual(p.get_latest(), 5)

    def test_settable(self):
        p = Parameter('p', set_cmd=self.set_p, get_cmd=False)

        p(10)
        self.assertEqual(self._p, 10)
        with self.assertRaises(NotImplementedError):
            p()

        self.assertTrue(hasattr(p, 'set'))
        self.assertFalse(hasattr(p, 'get'))

        # For settable-only parameters, using ``cache.set`` may not make
        # sense, nevertheless, it works
        p.cache.set(7)
        self.assertEqual(p.get_latest(), 7)

    def test_gettable(self):
        p = Parameter('p', get_cmd=self.get_p)
        self._p = 21

        self.assertEqual(p(), 21)
        self.assertEqual(p.get(), 21)

        with self.assertRaises(NotImplementedError):
            p(10)

        self.assertTrue(hasattr(p, 'get'))
        self.assertFalse(hasattr(p, 'set'))

        p.cache.set(7)
        self.assertEqual(p.get_latest(), 7)
        # Nothing has been passed to the "instrument" at ``cache.set``
        # call, hence the following assertions should hold
        self.assertEqual(self._p, 21)
        self.assertEqual(p(), 21)
        self.assertEqual(p.get_latest(), 21)

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

        self._p = 1  # for further testing

        p.cache.set('off')
        self.assertEqual(p.get_latest(), 'off')
        # Nothing has been passed to the "instrument" at ``cache.set``
        # call, hence the following assertions should hold
        self.assertEqual(self._p, 1)
        self.assertEqual(p(), 'on')
        self.assertEqual(p.get_latest(), 'on')

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

        p.cache.set('off')
        self.assertEqual(p.get_latest(), 'off')
        # Nothing has been passed to the "instrument" at ``cache.set``
        # call, hence the following assertions should hold
        self.assertEqual(self._p, 'PVAL: 1')
        self.assertEqual(p(), 'on')
        self.assertEqual(p.get_latest(), 'on')

    def test_on_off_val_mapping(self):
        instrument_value_for_on = 'on_'
        instrument_value_for_off = 'off_'

        parameter_return_value_for_on = True
        parameter_return_value_for_off = False

        p = Parameter('p', set_cmd=self.set_p, get_cmd=self.get_p,
                      val_mapping=create_on_off_val_mapping(
                          on_val=instrument_value_for_on,
                          off_val=instrument_value_for_off))

        test_data = [(instrument_value_for_on,
                      parameter_return_value_for_on,
                      ('On', 'on', 'ON', 1, True)),
                     (instrument_value_for_off,
                      parameter_return_value_for_off,
                      ('Off', 'off', 'OFF', 0, False))]

        for instr_value, parameter_return_value, inputs in test_data:
            for inp in inputs:
                # Setting parameter with any of the `inputs` is allowed
                p(inp)
                # For any value from the `inputs`, what gets send to the
                # instrument is on_val/off_val which are specified in
                # `create_on_off_val_mapping`
                self.assertEqual(self._p, instr_value)
                # When getting a value of the parameter, only specific
                # values are returned instead of `inputs`
                self.assertEqual(p(), parameter_return_value)


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


class TestScaledParameter(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent_instrument = DummyInstrument('dummy')

    def setUp(self):
        self.target_name = 'target_parameter'
        self.target_label = 'Target Parameter'
        self.target_unit = 'V'

        self.target = ManualParameter(name=self.target_name, label=self.target_label,
                                      unit=self.target_unit, initial_value=1.0,
                                      instrument=self.parent_instrument)
        self.parent_instrument.add_parameter(self.target)
        self.scaler = ScaledParameter(self.target, division=1)

    @classmethod
    def tearDownClass(cls):
        cls.parent_instrument.close()
        del cls.parent_instrument

    def test_constructor(self):
        #Test the behaviour of the constructor

        # Require a wrapped parameter
        with self.assertRaises(TypeError):
            ScaledParameter()

        # Require a scaling factor
        with self.assertRaises(ValueError):
            ScaledParameter(self.target)

        # Require only one scaling factor
        with self.assertRaises(ValueError):
            ScaledParameter(self.target, division=1, gain=1)

    def test_namelabel(self):
        #Test handling of name and label

        # Test correct inheritance
        assert self.scaler.name == self.target_name + '_scaled'
        assert self.scaler.label == self.target_label + '_scaled'

        # Test correct name/label handling by the constructor
        scaled_name = 'scaled'
        scaled_label = "Scaled parameter"
        scaler2 = ScaledParameter(self.target, division=1, name=scaled_name, label=scaled_label)
        assert scaler2.name == scaled_name
        assert scaler2.label == scaled_label

    def test_unit(self):
        # Test handling of the units

        # Check if the unit is correctly inherited
        assert self.scaler.unit == 'V'

        # Check if we can change succesfully the unit
        self.scaler.unit = 'A'
        assert self.scaler.unit == 'A'

        # Check if unit is correctly set in the constructor
        scaler2 = ScaledParameter(self.target, name='scaled_value', division=1, unit='K')
        assert scaler2.unit == 'K'

    def test_metadata(self):
        #Test the metadata

        test_gain = 3
        test_unit = 'V'
        self.scaler.gain = test_gain
        self.scaler.unit = test_unit

        # Check if relevant fields are present in the snapshot
        snap = self.scaler.snapshot()
        snap_keys = snap.keys()
        metadata_keys = snap['metadata'].keys()
        assert 'division' in snap_keys
        assert 'gain' in snap_keys
        assert 'role' in snap_keys
        assert 'unit' in snap_keys
        assert 'variable_multiplier' in metadata_keys
        assert 'wrapped_parameter' in metadata_keys
        assert 'wrapped_instrument' in metadata_keys

        # Check if the fields are correct
        assert snap['gain'] == test_gain
        assert snap['division'] == 1/test_gain
        assert snap['role'] == ScaledParameter.Role.GAIN
        assert snap['unit'] == test_unit
        assert snap['metadata']['variable_multiplier'] == False
        assert snap['metadata']['wrapped_parameter'] == self.target.name

    def test_wrapped_parameter(self):
        #Test if the target parameter is correctly inherited
        assert self.scaler.wrapped_parameter == self.target

    def test_divider(self):
        test_division = 10
        test_value = 5

        self.scaler.division = test_division
        self.scaler(test_value)
        assert self.scaler() == test_value
        assert self.target() == test_division * test_value
        assert self.scaler.gain == 1/test_division
        assert self.scaler.role == ScaledParameter.Role.DIVISION

    def test_multiplier(self):
        test_multiplier= 10
        test_value = 5

        self.scaler.gain = test_multiplier
        self.scaler(test_value)
        assert self.scaler() == test_value
        assert self.target() == test_value / test_multiplier
        assert self.scaler.division == 1/test_multiplier
        assert self.scaler.role == ScaledParameter.Role.GAIN

    def test_variable_gain(self):
        test_value = 5

        initial_gain = 2
        variable_gain_name = 'gain'
        gain = ManualParameter(name=variable_gain_name, initial_value=initial_gain)
        self.scaler.gain = gain
        self.scaler(test_value)

        assert self.scaler() == test_value
        assert self.target() == test_value / initial_gain
        assert self.scaler.division == 1/initial_gain

        second_gain = 7
        gain(second_gain)
        assert self.target() == test_value / initial_gain   #target value must change on scaler value change, not on gain/division
        self.scaler(test_value)
        assert self.target() == test_value / second_gain
        assert self.scaler.division == 1 / second_gain

        assert self.scaler.metadata['variable_multiplier'] == variable_gain_name


class TestSetContextManager(TestCase):

    def setUp(self):
        self.instrument = DummyInstrument('dummy_holder')

        self.instrument.add_parameter("a",
                                      set_cmd=None,
                                      get_cmd=None)

        # These two parameters mock actual instrument parameters; when first
        # connecting to the instrument, they have the _latest["value"] None.
        # We must call get() on them to get a valid value that we can set
        # them to in the __exit__ method of the context manager
        self.instrument.add_parameter("validated_param",
                                      set_cmd=self._vp_setter,
                                      get_cmd=self._vp_getter,
                                      vals=vals.Enum("foo", "bar"))

        self.instrument.add_parameter("parsed_param",
                                      set_cmd=self._pp_setter,
                                      get_cmd=self._pp_getter,
                                      set_parser=int)

        # A parameter that counts the number of times it has been set
        self.instrument.add_parameter("counting_parameter",
                                      set_cmd=self._cp_setter,
                                      get_cmd=self._cp_getter)

        # the mocked instrument state values of validated_param and
        # parsed_param
        self._vp_value = "foo"
        self._pp_value = 42

        # the counter value for counting_parameter
        self._cp_counter = 0
        self._cp_get_counter = 0

    def _vp_getter(self):
        return self._vp_value

    def _vp_setter(self, value):
        self._vp_value = value

    def _pp_getter(self):
        return self._pp_value

    def _pp_setter(self, value):
        self._pp_value = value

    def _cp_setter(self, value):
        self._cp_counter += 1

    def _cp_getter(self):
        self._cp_get_counter += 1
        return self.instrument['counting_parameter'].cache._value

    def tearDown(self):
        self.instrument.close()
        del self.instrument

    def test_set_to_none_when_parameter_is_not_captured_yet(self):
        counting_parameter = self.instrument.counting_parameter
        # Pre-conditions:
        assert self._cp_counter == 0
        assert self._cp_get_counter == 0
        assert counting_parameter.cache._value is None
        assert counting_parameter.get_latest.get_timestamp() is None

        with counting_parameter.set_to(None):
            # The value should not change
            assert counting_parameter.cache._value is None
            # The timestamp of the latest value should not be None anymore
            assert counting_parameter.get_latest.get_timestamp() is not None
            # Set method is not called
            assert self._cp_counter == 0
            # Get method is called once
            assert self._cp_get_counter == 1

        # The value should not change
        assert counting_parameter.cache._value is None
        # The timestamp of the latest value should still not be None
        assert counting_parameter.get_latest.get_timestamp() is not None
        # Set method is still not called
        assert self._cp_counter == 0
        # Get method is still called once
        assert self._cp_get_counter == 1

    def test_set_to_none_for_not_captured_parameter_but_instrument_has_value(self):
        # representing instrument here
        instr_value = 'something'
        set_counter = 0

        def set_instr_value(value):
            nonlocal instr_value, set_counter
            instr_value = value
            set_counter += 1

        # make a parameter that is linked to an instrument
        p = Parameter('p', set_cmd=set_instr_value, get_cmd=lambda: instr_value,
                      val_mapping={'foo': 'something', None: 'nothing'})

        # pre-conditions
        assert p.cache._value is None
        assert p.cache._raw_value is None
        assert p.cache.timestamp is None
        assert set_counter == 0

        with p.set_to(None):
            # assertions after entering the context
            assert set_counter == 1
            assert instr_value == 'nothing'
            assert p.cache._value is None
            assert p.cache._raw_value == 'nothing'
            assert p.cache.timestamp is not None

        # assertions after exiting the context
        assert set_counter == 2
        assert instr_value == 'something'
        assert p.cache._value == 'foo'
        assert p.cache._raw_value == 'something'
        assert p.cache.timestamp is not None

    def test_none_value(self):
        with self.instrument.a.set_to(3):
            assert self.instrument.a.get_latest.get_timestamp() is not None
            assert self.instrument.a.get() == 3
        assert self.instrument.a.get() is None
        assert self.instrument.a.get_latest.get_timestamp() is not None

    def test_context(self):
        self.instrument.a.set(2)

        with self.instrument.a.set_to(3):
            assert self.instrument.a.get() == 3
        assert self.instrument.a.get() == 2

    def test_validated_param(self):
        assert self.instrument.parsed_param.cache._value is None
        assert self.instrument.validated_param.get_latest() == "foo"
        with self.instrument.validated_param.set_to("bar"):
            assert self.instrument.validated_param.get() == "bar"
        assert self.instrument.validated_param.get_latest() == "foo"
        assert self.instrument.validated_param.get() == "foo"

    def test_parsed_param(self):
        assert self.instrument.parsed_param.cache._value is None
        assert self.instrument.parsed_param.get_latest() == 42
        with self.instrument.parsed_param.set_to(1):
            assert self.instrument.parsed_param.get() == 1
        assert self.instrument.parsed_param.get_latest() == 42
        assert self.instrument.parsed_param.get() == 42

    def test_number_of_set_calls(self):
        """
        Test that with param.set_to(X) does not perform any calls to set if
        the parameter already had the value X
        """
        assert self._cp_counter == 0
        self.instrument.counting_parameter(1)
        assert self._cp_counter == 1

        with self.instrument.counting_parameter.set_to(2):
            pass
        assert self._cp_counter == 3

        with self.instrument.counting_parameter.set_to(1):
            pass
        assert self._cp_counter == 3


def test_deprecated_param_warns():
    """
    Test that creating a parameter that has deprecated get and set still works
    but raises the correct warnings.
    """

    with pytest.warns(UserWarning) as record:
        a = DeprecatedParam(name='foo')
    assert len(record) == 2
    assert record[0].message.args[0] == ("Wrapping get method of parameter: "
                                         "foo, original get method will not be "
                                         "directly accessible. It is "
                                         "recommended to define get_raw in "
                                         "your subclass instead. Overwriting "
                                         "get will be an error in the future.")
    assert record[1].message.args[0] == ("Wrapping set method of parameter: "
                                         "foo, original set method will not be "
                                         "directly accessible. It is "
                                         "recommended to define set_raw in "
                                         "your subclass instead. Overwriting "
                                         "set will be an error in the future.")
    # test that get and set are called as expected (not shadowed by wrapper)
    assert a.get_count == 0
    assert a.set_count == 0
    assert a.get() == 42
    assert a.get_count == 1
    assert a.set_count == 0
    a.set(11)
    assert a.get_count == 1
    assert a.set_count == 1
    assert a.get() == 11
    assert a.get_count == 2
    assert a.set_count == 1
    # check that wrapper functionality works e.g stepping is performed
    # correctly
    a.step = 1
    a.set(20)
    assert a.set_count == 1+9
    assert a.get() == 20
    assert a.get_count == 3


def test_unknown_args_to_baseparameter_warns():
    """
    Passing an unknown kwarg to _BaseParameter should trigger a warning
    """
    with pytest.warns(Warning):
        a = _BaseParameter(name='Foo',
                           instrument=None,
                           snapshotable=False)
