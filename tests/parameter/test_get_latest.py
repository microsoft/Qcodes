import time
from datetime import datetime, timedelta
from typing import Any

import pytest

from qcodes.parameters import Parameter, ParameterBase

from .conftest import BetterGettableParam


def test_get_latest() -> None:
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
    assert local_parameter.get_latest() == 1
    get_timestamp = local_parameter.get_latest.get_timestamp()
    assert get_timestamp is not None
    assert before_set < get_timestamp < after_set

    # Check that updating the value updates the timestamp
    time.sleep(sleep_delta)
    local_parameter.set(2)
    get_timestamp = local_parameter.get_latest.get_timestamp()
    assert get_timestamp is not None
    assert local_parameter.get_latest() == 2
    assert get_timestamp > after_set


def test_get_latest_raw_value() -> None:
    # To have a simple distinction between raw value and value of the
    # parameter lets create a parameter with an offset
    p = Parameter("p", set_cmd=None, get_cmd=None, offset=42)
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


def test_get_latest_unknown() -> None:
    """
    Test that get latest on a parameter that has not been acquired will
    trigger a get
    """
    value = 1
    local_parameter = BetterGettableParam("test_param", set_cmd=None, get_cmd=None)
    # fake a parameter that has a value but never been get/set to mock
    # an instrument.
    local_parameter.cache._value = value  # type: ignore[attr-defined]
    local_parameter.cache._raw_value = value  # type: ignore[attr-defined]
    assert local_parameter.get_latest.get_timestamp() is None
    before_get = datetime.now()
    assert local_parameter._get_count == 0
    assert local_parameter.get_latest() == value
    assert local_parameter._get_count == 1
    # calling get_latest above will call get since TS is None
    # and the TS will therefore no longer be None
    get_timestamp = local_parameter.get_latest.get_timestamp()
    assert get_timestamp is not None
    assert get_timestamp >= before_get
    # calling get_latest now will not trigger get
    assert local_parameter.get_latest() == value
    assert local_parameter._get_count == 1


def test_get_latest_known() -> None:
    """
    Test that get latest on a parameter that has a known value will not
    trigger a get
    """
    value = 1
    local_parameter = BetterGettableParam("test_param", set_cmd=None, get_cmd=None)
    # fake a parameter that has a value acquired 10 sec ago
    start = datetime.now()
    set_time = start - timedelta(seconds=10)
    local_parameter.cache._update_with(value=value, raw_value=value, timestamp=set_time)
    assert local_parameter._get_count == 0
    assert local_parameter.get_latest.get_timestamp() == set_time
    assert local_parameter.get_latest() == value
    # calling get_latest above will not call get since TS is set and
    # max_val_age is not
    assert local_parameter._get_count == 0
    assert local_parameter.get_latest.get_timestamp() == set_time


def test_get_latest_no_get() -> None:
    """
    Test that get_latest on a parameter that does not have get is handled
    correctly.
    """
    local_parameter = Parameter("test_param", set_cmd=None, get_cmd=False)
    # The parameter does not have a get method.
    with pytest.raises(AttributeError):
        local_parameter.get()
    # get_latest will fail as get cannot be called and no cache
    # is available
    with pytest.raises(RuntimeError):
        local_parameter.get_latest()
    value = 1
    local_parameter.set(value)
    assert local_parameter.get_latest() == value

    local_parameter2 = Parameter(
        "test_param2", set_cmd=None, get_cmd=False, initial_value=value
    )
    with pytest.raises(AttributeError):
        local_parameter2.get()
    assert local_parameter2.get_latest() == value


def test_max_val_age() -> None:
    value = 1
    start = datetime.now()
    local_parameter = BetterGettableParam(
        "test_param", set_cmd=None, max_val_age=1, initial_value=value
    )
    assert local_parameter.cache.max_val_age == 1
    assert local_parameter._get_count == 0
    assert local_parameter.get_latest() == value
    assert local_parameter._get_count == 0
    # now fake the time stamp so get should be triggered
    set_time = start - timedelta(seconds=10)
    local_parameter.cache._update_with(value=value, raw_value=value, timestamp=set_time)
    # now that ts < max_val_age calling get_latest should update the time
    assert local_parameter.get_latest.get_timestamp() == set_time
    assert local_parameter.get_latest() == value
    assert local_parameter._get_count == 1
    new_get_time_stamp = local_parameter.get_latest.get_timestamp()
    assert new_get_time_stamp is not None
    assert new_get_time_stamp >= start


def test_no_get_max_val_age() -> None:
    """
    Test that get_latest on a parameter with max_val_age set and
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

    # ParameterBase does not have this check on creation time since get_cmd
    # could be added in a subclass. Here we create a subclass that does add a
    # get command and also does not implement the check for max_val_age
    class LocalParameter(ParameterBase):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.set_raw = lambda x: x  # type: ignore[method-assign]
            self.set = self._wrap_set(self.set_raw)

    localparameter = LocalParameter("test_param", None, max_val_age=1)
    with pytest.raises(RuntimeError):
        localparameter.get_latest()
