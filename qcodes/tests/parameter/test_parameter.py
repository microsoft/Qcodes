"""
Test suite for parameter
"""
from collections.abc import Iterable
from unittest import TestCase
import pytest


import numpy as np
from hypothesis import given, event, settings
import hypothesis.strategies as hst
from qcodes import Function
from qcodes.instrument.parameter import Parameter, _BaseParameter
import qcodes.utils.validators as vals
from qcodes.tests.instrument_mocks import DummyInstrument
from qcodes.utils.helpers import create_on_off_val_mapping
from qcodes.utils.validators import Numbers
from .conftest import (MemoryParameter,
                       OverwriteGetParam, OverwriteSetParam,
                       GetSetRawParameter)


class TestParameter(TestCase):

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

    if not p.gettable:
        assert not hasattr(p, 'get')
        assert p.gettable is False
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
        self.assertTrue(p.settable)
        self.assertFalse(hasattr(p, 'get'))
        self.assertFalse(p.gettable)

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
        self.assertTrue(p.gettable)
        self.assertFalse(hasattr(p, 'set'))
        self.assertFalse(p.settable)

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


def test_parameter_with_overwritten_get_raises():
    """
    Test that creating a parameter that overwrites get and set raises runtime errors
    """

    with pytest.raises(RuntimeError) as record:
        a = OverwriteGetParam(name='foo')
    assert "Overwriting get in a subclass of _BaseParameter: foo is not allowed." == str(record.value)


def test_parameter_with_overwritten_set_raises():
    """
    Test that creating a parameter that overwrites get and set raises runtime errors
    """
    with pytest.raises(RuntimeError) as record:
        a = OverwriteSetParam(name='foo')
    assert "Overwriting set in a subclass of _BaseParameter: foo is not allowed." == str(record.value)


def test_unknown_args_to_baseparameter_warns():
    """
    Passing an unknown kwarg to _BaseParameter should trigger a warning
    """
    with pytest.warns(Warning):
        a = _BaseParameter(name='Foo',
                           instrument=None,
                           snapshotable=False)


@pytest.mark.parametrize("get_cmd, set_cmd", [(False, False), (False, None), (None, None), (None, False),
                                              (lambda: 1, lambda x: x)])
def test_gettable_settable_attributes_with_get_set_cmd(get_cmd, set_cmd):
    a = Parameter(name='foo',
                  get_cmd=get_cmd,
                  set_cmd=set_cmd)
    expected_gettable = get_cmd is not False
    expected_settable = set_cmd is not False

    assert a.gettable is expected_gettable
    assert a.settable is expected_settable


@pytest.mark.parametrize("baseclass", [_BaseParameter, Parameter])
def test_gettable_settable_attributes_with_get_set_raw(baseclass):
    """Test that parameters that have get_raw,set_raw are
    listed as gettable/settable and reverse."""

    class GetSetParam(baseclass):
        def __init__(self, *args, initial_value=None, **kwargs):
            self._value = initial_value
            super().__init__(*args, **kwargs)

        def get_raw(self):
            return self._value

        def set_raw(self, value):
            self._value = value

    a = GetSetParam('foo', instrument=None, initial_value=1)

    assert a.gettable is True
    assert a.settable is True

    b = _BaseParameter('foo', None)

    assert b.gettable is False
    assert b.settable is False


@pytest.mark.parametrize("working_get_cmd", (False, None))
@pytest.mark.parametrize("working_set_cmd", (False, None))
def test_get_raw_and_get_cmd_raises(working_get_cmd, working_set_cmd):
    with pytest.raises(TypeError, match="get_raw"):
        GetSetRawParameter(name="param1", get_cmd="GiveMeTheValue", set_cmd=working_set_cmd)
    with pytest.raises(TypeError, match="set_raw"):
        GetSetRawParameter(name="param2", set_cmd="HereIsTheValue {}", get_cmd=working_get_cmd)
    GetSetRawParameter("param3", get_cmd=working_get_cmd, set_cmd=working_set_cmd)


def test_get_on_parameter_marked_as_non_gettable_raises():
    a = Parameter("param")
    a._gettable = False
    with pytest.raises(TypeError, match="Trying to get a parameter that is not gettable."):
        a.get()


def test_set_on_parameter_marked_as_non_settable_raises():
    a = Parameter("param", set_cmd=None)
    a.set(2)
    assert a.get() == 2
    a._settable = False
    with pytest.raises(TypeError, match="Trying to set a parameter that is not settable."):
        a.set(1)
    assert a.get() == 2
