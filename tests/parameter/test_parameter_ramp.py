import logging

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import given, settings
from pytest import LogCaptureFixture

from qcodes.parameters import Parameter
from qcodes.validators import Ints, Numbers

from .conftest import MemoryParameter


def test_step_ramp(caplog: LogCaptureFixture) -> None:
    p = MemoryParameter(name="test_step")
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


@given(scale=hst.integers(1, 100), value=hst.floats(min_value=1e-9, max_value=10))
def test_ramp_scaled(scale: int, value: float) -> None:
    start_point = 0.0
    p = MemoryParameter(name="p", scale=scale, initial_value=start_point)
    assert p() == start_point
    # first set a step size
    p.step = 0.1
    # and a wait time
    p.inter_delay = 1e-9  # in seconds
    first_step = 1.0
    second_step = 10.0
    # do a step to start from a non zero starting point where
    # scale matters
    p.set(first_step)
    np.testing.assert_allclose(np.array([p.get()]), np.array([first_step]))

    expected_raw_steps = np.linspace(start_point * scale, first_step * scale, 11)
    # getting the raw values that are actually send to the instrument.
    # these are scaled in the set_wrapper
    np.testing.assert_allclose(np.array(p.set_values), expected_raw_steps)
    assert p.raw_value == first_step * scale
    # then check the generated steps. They should not be scaled as the
    # scaling happens when setting them
    expected_steps = np.linspace(first_step + p.step, second_step, 90)
    actual_steps = p.get_ramp_values(second_step, p.step)
    np.testing.assert_allclose(np.array(actual_steps), expected_steps)
    p.set(10)
    np.testing.assert_allclose(
        np.array(p.set_values), np.linspace(0.0 * scale, 10 * scale, 101)
    )
    p.set(value)
    np.testing.assert_allclose(p.get(), value)
    assert p.raw_value == value * scale


@given(value=hst.floats(min_value=1e-9, max_value=10))
def test_ramp_parser(value: float) -> None:
    start_point = 0.0
    p = MemoryParameter(
        name="p",
        set_parser=lambda x: -x,
        get_parser=lambda x: -x,
        initial_value=start_point,
    )
    assert p() == start_point
    # first set a step size
    p.step = 0.1
    # and a wait time
    p.inter_delay = 1e-9  # in seconds
    first_step = 1.0
    second_step = 10.0
    # do a step to start from a non zero starting point where
    # scale matters
    p.set(first_step)
    assert p.get() == first_step
    assert p.raw_value == -first_step
    np.testing.assert_allclose(np.array([p.get()]), np.array([first_step]))

    expected_raw_steps = np.linspace(-start_point, -first_step, 11)
    # getting the raw values that are actually send to the instrument.
    # these are parsed in the set_wrapper
    np.testing.assert_allclose(np.array(p.set_values), expected_raw_steps)
    assert p.raw_value == -first_step
    # then check the generated steps. They should not be parsed as the
    # scaling happens when setting them
    expected_steps = np.linspace((first_step + p.step), second_step, 90)
    actual_steps = p.get_ramp_values(second_step, p.step)
    np.testing.assert_allclose(np.array(actual_steps), expected_steps)
    p.set(second_step)
    np.testing.assert_allclose(
        np.array(p.set_values), np.linspace(-start_point, -second_step, 101)
    )
    p.set(value)
    np.testing.assert_allclose(p.get(), value)
    assert p.raw_value == -value


@given(scale=hst.integers(1, 100), value=hst.floats(min_value=1e-9, max_value=10))
@settings(deadline=None)
def test_ramp_parsed_scaled(scale: int, value: float) -> None:
    start_point = 0.0
    p = MemoryParameter(
        name="p",
        scale=scale,
        set_parser=lambda x: -x,
        get_parser=lambda x: -x,
        initial_value=start_point,
    )
    assert p() == start_point
    # first set a step size
    p.step = 0.1
    # and a wait time
    p.inter_delay = 1e-9  # in seconds
    first_step = 1.0
    second_step = 10.0
    p.set(first_step)
    assert p.get() == first_step
    assert p.raw_value == -first_step * scale
    expected_raw_steps = np.linspace(-start_point * scale, -first_step * scale, 11)
    # getting the raw values that are actually send to the instrument.
    # these are parsed in the set_wrapper
    np.testing.assert_allclose(np.array(p.set_values), expected_raw_steps)
    assert p.raw_value == -scale * first_step
    expected_steps = np.linspace(first_step + p.step, second_step, 90)
    actual_steps = p.get_ramp_values(10, p.step)
    np.testing.assert_allclose(np.array(actual_steps), expected_steps)
    p.set(second_step)
    np.testing.assert_allclose(
        np.array(p.set_values),
        np.linspace(-start_point * scale, -second_step * scale, 101),
    )
    p.set(value)
    np.testing.assert_allclose(p.get(), value)
    assert p.raw_value == -scale * value


@pytest.mark.parametrize(
    "target_type", [np.int32, np.int64, np.float32, np.float64, int, float]
)
def test_step_ramp_numpy_scalar_target(
    target_type: type, caplog: LogCaptureFixture
) -> None:
    # regression test for https://github.com/microsoft/Qcodes/issues/7203
    p = MemoryParameter(name="p", initial_value=0.0)
    p.step = 1
    with caplog.at_level(logging.WARNING):
        p(target_type(3))
        # set back with a different type so the cached start value is a
        # numpy scalar
        p(2.0)
    assert not caplog.records
    np.testing.assert_allclose(p.set_values, [0, 1, 2, 3, 2])


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16]
)
def test_step_ramp_numpy_integer_down(dtype: type[np.integer]) -> None:
    # stepping down must not wrap around for (unsigned) numpy integers
    p = MemoryParameter(name="p", initial_value=dtype(100))
    p.step = 25
    p(dtype(0))
    assert p.set_values == [100, 75, 50, 25, 0]
    p(dtype(100))
    assert p.set_values == [100, 75, 50, 25, 0, 25, 50, 75, 100]


def test_step_ramp_numpy_integer_overflow() -> None:
    p = MemoryParameter(name="p", initial_value=np.int8(100))
    p.step = 50
    p(np.int8(-100))
    assert p.set_values == [100, 50, 0, -50, -100]


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_step_ramp_numpy_integer_full_span(dtype: type[np.integer]) -> None:
    # ramping between the limits of the dtype must write every intermediate
    # step, never more than step apart, in both directions
    low, high = int(np.iinfo(dtype).min), int(np.iinfo(dtype).max)
    step = (high - low) // 5 + 1
    p = MemoryParameter(
        name="p", initial_value=dtype(low), vals=Ints(min_value=low, max_value=high)
    )
    p.step = step
    p(dtype(high))
    p(dtype(low))
    up = [*range(low, high, step), high]
    down = [*range(high, low, -step), low]
    assert p.set_values == up + down[1:]
    assert p() == low


def test_stepping_from_invalid_starting_point() -> None:
    the_value = -10

    def set_function(value: float) -> None:
        nonlocal the_value
        the_value = value

    def get_function() -> float:
        return the_value

    a = Parameter(
        "test", set_cmd=set_function, get_cmd=get_function, vals=Numbers(0, 100), step=5
    )
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
