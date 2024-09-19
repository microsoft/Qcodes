from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import pytest

import qcodes.validators as vals
from qcodes.parameters import Parameter, ParameterBase

from .conftest import NOT_PASSED, BetterGettableParam, SettableParam

if TYPE_CHECKING:
    from qcodes.instrument_drivers.mock_instruments import DummyChannelInstrument


def test_get_from_cache_does_not_trigger_real_get_if_get_if_invalid_false() -> None:
    """
    assert that calling get on the cache with get_if_invalid=False does
    not trigger a get of the parameter when parameter
    has expired due to max_val_age
    """
    param = BetterGettableParam(name="param", max_val_age=1)
    param.get()
    assert param._get_count == 1
    # let the cache expire
    time.sleep(2)
    param.cache.get(get_if_invalid=False)
    assert param._get_count == 1


def test_initial_set_with_without_cache() -> None:
    value = 43
    # setting the initial value triggers a set
    param1 = SettableParam(name="param", initial_value=value)
    assert param1._set_count == 1
    assert param1.cache.get(get_if_invalid=False) == value
    # setting the cache does not trigger a set
    param2 = SettableParam(name="param", initial_cache_value=value)
    assert param2._set_count == 0
    assert param2.cache.get(get_if_invalid=False) == value


def test_set_initial_and_initial_cache_raises() -> None:
    with pytest.raises(SyntaxError, match="`initial_value` and `initial_cache_value`"):
        Parameter(name="param", initial_value=1, initial_cache_value=2)


def test_get_cache() -> None:
    time_resolution = time.get_clock_info("time").resolution
    sleep_delta = 2 * time_resolution

    # Create a gettable parameter
    local_parameter = Parameter("test_param", set_cmd=None, get_cmd=None)
    before_set = datetime.now()
    time.sleep(sleep_delta)
    local_parameter.set(1)
    time.sleep(sleep_delta)
    after_set = datetime.now()

    # Check we return last set value, with the correct timestamp
    assert local_parameter.cache.get() == 1
    before_timestamp = local_parameter.cache.timestamp
    assert before_timestamp is not None
    assert before_set < before_timestamp < after_set

    # Check that updating the value updates the timestamp
    time.sleep(sleep_delta)
    local_parameter.set(2)
    assert local_parameter.cache.get() == 2
    after_timestamp = local_parameter.cache.timestamp
    assert after_timestamp is not None
    assert after_timestamp > after_set


def test_get_cache_raw_value() -> None:
    # To have a simple distinction between raw value and value of the
    # parameter lets create a parameter with an offset
    p = Parameter("p", set_cmd=None, get_cmd=None, offset=42)
    assert p.cache.timestamp is None

    # Initially, the parameter's raw value is None
    assert p.cache.raw_value is None

    # After setting the parameter to some value, the
    # raw_value attribute of the cache should return the raw_value
    p(3)
    assert p.cache.timestamp is not None
    assert p.cache.get() == 3
    assert p.cache.raw_value == 3 + 42


def test_get_cache_unknown() -> None:
    """
    Test that cache get on a parameter that has not been acquired will
    trigger a get
    """
    value = 1
    local_parameter = BetterGettableParam("test_param", set_cmd=None, get_cmd=None)
    # fake a parameter that has a value but never been get/set to mock
    # an instrument.
    local_parameter.cache._value = value  # type: ignore[attr-defined]
    local_parameter.cache._raw_value = value  # type: ignore[attr-defined]
    assert local_parameter.cache.timestamp is None
    before_get = datetime.now()
    assert local_parameter._get_count == 0
    assert local_parameter.cache.get() == value
    assert local_parameter._get_count == 1
    # calling get above will call get since TS is None
    # and the TS will therefore no longer be None
    assert local_parameter.cache.timestamp is not None
    assert local_parameter.cache.timestamp >= before_get
    # calling cache.get now will not trigger get
    assert local_parameter.cache.get() == value
    assert local_parameter._get_count == 1


def test_get_cache_known() -> None:
    """
    Test that cache.get on a parameter that has a known value will not
    trigger a get
    """
    value = 1
    local_parameter = BetterGettableParam("test_param", set_cmd=None, get_cmd=None)
    # fake a parameter that has a value acquired 10 sec ago
    start = datetime.now()
    set_time = start - timedelta(seconds=10)
    local_parameter.cache._update_with(value=value, raw_value=value, timestamp=set_time)
    assert local_parameter._get_count == 0
    assert local_parameter.cache.timestamp == set_time
    assert local_parameter.cache.get() == value
    # calling cache.get above will not call get since TS is set and
    # max_val_age is not
    assert local_parameter._get_count == 0
    assert local_parameter.cache.timestamp == set_time


def test_get_cache_no_get() -> None:
    """
    Test that cache.get on a parameter that does not have get is handled
    correctly.
    """
    local_parameter = Parameter("test_param", set_cmd=None, get_cmd=False)
    # The parameter does not have a get method.
    with pytest.raises(AttributeError):
        local_parameter.get()
    # get_latest will fail as get cannot be called and no cache
    # is available
    with pytest.raises(RuntimeError):
        local_parameter.cache.get()
    value = 1
    local_parameter.set(value)
    assert local_parameter.cache.get() == value

    local_parameter2 = Parameter(
        "test_param2", set_cmd=None, get_cmd=False, initial_value=value
    )
    with pytest.raises(AttributeError):
        local_parameter2.get()
    assert local_parameter2.cache.get() == value


def test_set_raw_value_on_cache() -> None:
    value = 1
    scale = 10
    local_parameter = BetterGettableParam("test_param", set_cmd=None, scale=scale)
    before = datetime.now()
    local_parameter.cache._set_from_raw_value(value * scale)
    after = datetime.now()
    assert local_parameter.cache.get(get_if_invalid=False) == value
    assert local_parameter.cache.raw_value == value * scale
    timestamp = local_parameter.cache.timestamp
    assert timestamp is not None
    assert timestamp >= before
    assert timestamp <= after


def test_max_val_age() -> None:
    value = 1
    start = datetime.now()
    local_parameter = BetterGettableParam(
        "test_param", set_cmd=None, max_val_age=1, initial_value=value
    )
    assert local_parameter.cache.max_val_age == 1
    assert local_parameter._get_count == 0
    assert local_parameter.cache.get() == value
    assert local_parameter._get_count == 0
    # now fake the time stamp so get should be triggered
    set_time = start - timedelta(seconds=10)
    local_parameter.cache._update_with(value=value, raw_value=value, timestamp=set_time)
    # now that ts < max_val_age calling get_latest should update the time
    assert local_parameter.cache.timestamp == set_time
    assert local_parameter.cache.get() == value
    assert local_parameter._get_count == 1
    timestamp = local_parameter.cache.timestamp
    assert timestamp is not None
    assert timestamp >= start


def test_no_get_max_val_age() -> None:
    """
    Test that cache.get on a parameter with max_val_age set and
    no get cmd raises correctly.
    """
    value = 1
    with pytest.raises(SyntaxError):
        _ = Parameter(
            "test_param",
            set_cmd=None,
            get_cmd=False,
            max_val_age=1,
            initial_value=value,
        )


def test_no_get_max_val_age_runtime_error(
    get_if_invalid: bool | Literal["NOT_PASSED"],
) -> None:
    """
    ParameterBase does not have a check on creation time that
    no get_cmd is mixed with max_val_age since get_cmd could be added
    in a subclass. Here we create a subclass that does not add a get
    command and also does not implement the check for max_val_age
    """
    value = 1

    class LocalParameter(ParameterBase):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.set_raw = lambda x: x  # type: ignore[method-assign]
            self.set = self._wrap_set(self.set_raw)

    local_parameter = LocalParameter("test_param", None, max_val_age=1)
    start = datetime.now()
    set_time = start - timedelta(seconds=10)
    local_parameter.cache._update_with(value=value, raw_value=value, timestamp=set_time)

    if get_if_invalid is True:
        with pytest.raises(RuntimeError, match="max_val_age` is not supported"):
            local_parameter.cache.get(get_if_invalid=get_if_invalid)
    elif get_if_invalid == NOT_PASSED:
        with pytest.raises(RuntimeError, match="max_val_age` is not supported"):
            local_parameter.cache.get()
    else:
        assert local_parameter.cache.get(get_if_invalid=get_if_invalid) == 1


def test_no_get_timestamp_none_runtime_error(
    get_if_invalid: bool | Literal["NOT_PASSED"],
) -> None:
    """
    Test that a parameter that has never been
    set, cannot be get and does not support
    getting raises a RuntimeError.
    """
    local_parameter = Parameter("test_param", get_cmd=False)

    if get_if_invalid is True:
        with pytest.raises(RuntimeError, match="Value of parameter test_param"):
            local_parameter.cache.get(get_if_invalid=get_if_invalid)
    elif get_if_invalid == NOT_PASSED:
        with pytest.raises(RuntimeError, match="Value of parameter test_param"):
            local_parameter.cache.get()
    else:
        assert local_parameter.cache.get(get_if_invalid=get_if_invalid) is None


def test_latest_dictionary_gets_updated_upon_set_of_memory_parameter() -> None:
    p = Parameter("p", set_cmd=None, get_cmd=None)
    assert p.cache._value is None  # type: ignore[attr-defined]
    assert p.cache.raw_value is None
    assert p.cache.timestamp is None

    p(42)

    assert p.cache._value == 42  # type: ignore[attr-defined]
    assert p.cache.raw_value == 42
    assert p.cache.timestamp is not None


_P = Parameter


@pytest.mark.parametrize(
    argnames=("p", "value", "raw_value"),
    argvalues=(
        (_P("p", set_cmd=None, get_cmd=None), 4, 4),
        (_P("p", set_cmd=False, get_cmd=None), 14, 14),
        (_P("p", set_cmd=None, get_cmd=False), 14, 14),
        (_P("p", set_cmd=None, get_cmd=None, vals=vals.OnOff()), "on", "on"),
        (_P("p", set_cmd=None, get_cmd=None, val_mapping={"screw": 1}), "screw", 1),
        (_P("p", set_cmd=None, get_cmd=None, set_parser=str, get_parser=int), 14, "14"),
        (_P("p", set_cmd=None, get_cmd=None, step=7), 14, 14),
        (_P("p", set_cmd=None, get_cmd=None, offset=3), 14, 17),
        (_P("p", set_cmd=None, get_cmd=None, scale=2), 14, 28),
        (_P("p", set_cmd=None, get_cmd=None, offset=-3, scale=2), 14, 25),
    ),
    ids=(
        "with_nothing_extra",
        "without_set_cmd",
        "without_get_cmd",
        "with_on_off_validator",
        "with_val_mapping",
        "with_set_and_parsers",
        "with_step",
        "with_offset",
        "with_scale",
        "with_scale_and_offset",
    ),
)
def test_set_latest_works_for_plain_memory_parameter(
    p: Parameter, value: str | int, raw_value: str | int
) -> None:
    # Set latest value of the parameter
    p.cache.set(value)

    # Assert the latest value and raw_value
    assert p.get_latest() == value
    assert p.raw_value == raw_value

    # Assert latest value and raw_value via private attributes for strictness
    assert p.cache._value == value  # type: ignore[attr-defined]
    assert p.cache.raw_value == raw_value

    # Now let's get the value of the parameter to ensure that the value that
    # is set above gets picked up from the `_latest` dictionary (due to
    # `get_cmd=None`)

    if not p.gettable:
        assert not hasattr(p, "get")
        assert p.gettable is False
        return  # finish the test here for non-gettable parameters

    gotten_value = p.get()

    assert gotten_value == value

    # Assert the latest value and raw_value
    assert p.get_latest() == value
    assert p.raw_value == raw_value

    # Assert latest value and raw_value via private attributes for strictness
    assert p.cache._value == value  # type: ignore[attr-defined]
    assert p.cache.raw_value == raw_value


def test_get_from_cache_marked_invalid() -> None:
    param = BetterGettableParam(name="param")
    param.get()
    assert param._get_count == 1

    param.cache.get(get_if_invalid=False)
    assert param._get_count == 1

    param.cache.invalidate()

    param.cache.get(get_if_invalid=True)
    assert param._get_count == 2

    param.cache.invalidate()

    param.cache.get(get_if_invalid=False)
    assert param._get_count == 2

    param._gettable = False

    with pytest.raises(
        RuntimeError,
        match="Cannot return cache of a parameter"
        " that does not have a get command"
        " and has an invalid cache",
    ):
        param.cache.get(get_if_invalid=True)
    assert param._get_count == 2


def test_marking_invalid_via_instrument(
    dummy_instrument: DummyChannelInstrument,
) -> None:
    def _assert_cache_status(valid: bool) -> None:
        for param in dummy_instrument.parameters.values():
            assert param.cache.valid is valid, param.full_name

        for instrument_module in dummy_instrument.instrument_modules.values():
            for param in instrument_module.parameters.values():
                # parameters not snapshotted will not have a cache
                # updated when calling snapshot(update=None) os
                # exclude them
                if (
                    param._snapshot_get is True
                    and param.snapshot_value is True
                    and param.snapshot_exclude is False
                ):
                    assert param.cache.valid is valid, param.full_name

    dummy_instrument.snapshot(update=None)

    _assert_cache_status(True)

    dummy_instrument.invalidate_cache()

    _assert_cache_status(False)

    dummy_instrument.snapshot(update=None)

    _assert_cache_status(True)

    dummy_instrument.invalidate_cache()

    _assert_cache_status(False)

    for param in dummy_instrument.parameters.values():
        param.get()
    for module in dummy_instrument.instrument_modules.values():
        for param in module.parameters.values():
            param.get()

    _assert_cache_status(True)
