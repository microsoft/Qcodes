import logging

import pytest
import hypothesis.strategies as hst
from hypothesis import given, settings
import numpy as np

from .conftest import MemoryParameter
from qcodes.utils.validators import Numbers
from qcodes.instrument.parameter import Parameter


def test_step_ramp(caplog):
    p = MemoryParameter(name='test_step')
    p(42)
    assert p.set_values == [42]
    p.step = 1

    assert p.get_ramp_values(44.5, 1) == [43, 44, 44.5]

    p(44.5)
    assert p.set_values == [42, 43, 44, 44.5]

    # Assert that stepping does not impact ``cache.set`` call, and that
    # the value that is passed to ``cache.set`` call does not get
    # propagated to parameter's ``set_cmd``
    p.cache.set(40)
    assert p.get_latest() == 40
    assert p.set_values == [42, 43, 44, 44.5]

    # Test error conditions
    with caplog.at_level(logging.WARNING):
        assert p.get_ramp_values("A", 1) == ["A"]
        assert len(caplog.records) == 1
        assert "cannot sweep test_step from 40 to 'A'" in str(caplog.records[0])
    with pytest.raises(RuntimeError):
        p.get_ramp_values((1, 2, 3), 1)


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
@settings(deadline=None)
def test_ramp_parsed_scaled(scale, value):
    start_point = 0.0
    p = MemoryParameter(name='p',
                        scale=scale,
                        set_parser=lambda x: -x,
                        get_parser=lambda x: -x,
                        initial_value=start_point)
    assert p() == start_point
    # first set a step size
    p.step = 0.1
    # and a wait time
    p.inter_delay = 1e-9  # in seconds
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
    expected_steps = np.linspace(first_step+p.step, second_step, 90)
    np.testing.assert_allclose(p.get_ramp_values(10, p.step),
                               expected_steps)
    p.set(second_step)
    np.testing.assert_allclose(np.array(p.set_values),
                               np.linspace(-start_point*scale,
                                           -second_step*scale, 101))
    p.set(value)
    np.testing.assert_allclose(p.get(), value)
    assert p.raw_value == -scale * value


def test_stepping_from_invalid_starting_point():

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
