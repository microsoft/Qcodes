from collections.abc import Iterable

import hypothesis.strategies as hst
from hypothesis import given, event, settings
import numpy as np
import pytest

from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import Numbers
from .conftest import MemoryParameter


def test_scale_raw_value():
    p = Parameter(name='test_scale_raw_value', set_cmd=None)
    p(42)
    assert p.raw_value == 42

    p.scale = 2
    assert p.raw_value == 42  # No set/get cmd performed
    assert p() == 21

    p(10)
    assert p.raw_value == 20
    assert p() == 10


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
def test_scale_and_offset_raw_value_iterable(values, offsets, scales):
    p = Parameter(name='test_scale_and_offset_raw_value', set_cmd=None)

    # test that scale and offset does not change the default behaviour
    p(values)
    assert p.raw_value == values

    # test setting scale and offset does not change anything
    p.scale = scales
    p.offset = offsets
    assert p.raw_value == values

    np_values = np.array(values)
    np_offsets = np.array(offsets)
    np_scales = np.array(scales)
    np_get_values = np.array(p())
    np.testing.assert_allclose(np_get_values, (np_values - np_offsets) / np_scales)  # No set/get cmd performed

    # test set, only for scalar values
    if not isinstance(values, Iterable):
        p(values)
        np.testing.assert_allclose(np.array(p.raw_value),
                                   np_values * np_scales + np_offsets)  # No set/get cmd performed

        # testing conversion back and forth
        p(values)
        np_get_values = np.array(p())
        np.testing.assert_allclose(np_get_values, np_values)  # No set/get cmd performed

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
def test_scale_and_offset_raw_value_iterable_for_set_cache(values,
                                                           offsets,
                                                           scales):
    p = Parameter(name='test_scale_and_offset_raw_value', set_cmd=None)

    # test that scale and offset does not change the default behaviour
    p.cache.set(values)
    assert p.raw_value == values

    # test setting scale and offset does not change anything
    p.scale = scales
    p.offset = offsets
    assert p.raw_value == values

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
def test_ramp_scaled(scale, value):
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
def test_ramp_parser(value):
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
def test_ramp_parsed_scaled(scale, value):
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


def test_steppeing_from_invalid_starting_point():

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