import pytest

import qcodes.validators as vals

from qcodes_contrib_drivers.drivers.HP.private.ParameterMixinBase import OnValueChangeParameterMixin, mixin_parameter_factory
OnValueChangeParameter = mixin_parameter_factory(OnValueChangeParameterMixin)

# from qcodes_contrib_drivers.drivers.HP.private.ParameterMixinBase import OnValueChangeParameterMixin, mixin_parameter_factory
# OnValueChangeParameter = mixin_parameter_factory(OnValueChangeParameterMixin)

def test_on_value_change_source_is_correct() -> None:
    """
    Test that the correct 'source' value ('get_raw' or 'set_raw') is passed to the callback.
    """
    # List to track callback invocations and arguments
    callback_calls = []

    # Mock `get_cmd` value
    get_cmd_val = None

    def callback(old_value, new_value, source):
        callback_calls.append((old_value, new_value, source))

    # Create a parameter with the on_value_change callback
    p = OnValueChangeParameter(
        name="p",
        set_cmd=None,
        get_cmd=lambda: get_cmd_val,
        vals=vals.Anything(),
        on_value_change=callback,
    )

    # Test `set_raw` behavior
    p.set(10)  # Trigger `set_raw`
    assert callback_calls == [(None, 10, "set_raw")]

    # Reset callback_calls
    callback_calls.clear()

    # Test `get_raw` behavior
    get_cmd_val = 20
    p.get()  # Trigger `get_raw`
    assert callback_calls == [(10, 20, "get_raw")]

    # Reset callback_calls
    callback_calls.clear()

    # Test another `set_raw` update
    p.set(30)
    assert callback_calls == [(20, 30, "set_raw")]

    # Reset callback_calls
    callback_calls.clear()

    # Test another `get_raw` update
    get_cmd_val = 40
    p.get()
    assert callback_calls == [(30, 40, "get_raw")]


def test_parameter_on_value_change() -> None:
    """
    Test that the on_value_change callback is called appropriately when the parameter's value changes.
    """

    # List to track callback invocations and arguments
    callback_calls = []

    get_cmd_val = None

    def callback(old_value, new_value, source):
        callback_calls.append((old_value, new_value, source))

    # Create a parameter with the on_value_change callback
    p = OnValueChangeParameter(
        name="p",
        set_cmd=None,
        get_cmd=lambda: get_cmd_val,
        vals=vals.Anything(), 
        on_value_change=callback,
    )

    # Callback should not be called when parameter instance is created.
    assert callback_calls == []

    # Initial state: value is None
    assert p.get() is None

    # Set a new value (from None to 10)
    p.set(10)
    assert callback_calls == [(None, 10, 'set')]

    # Reset callback_calls
    callback_calls.clear()

    # Set the same value again; callback should not be called
    p.set(10)
    assert callback_calls == []

    # Set a different value (from 10 to 20)
    p.set(20)
    assert callback_calls == [(10, 20, 'set')]

    # Reset callback_calls
    callback_calls.clear()

    # Set the value back to None (from 20 to None)
    p.set(None)
    assert callback_calls == [(20, None, 'set')]

    # Reset callback_calls
    callback_calls.clear()

    # Set to the same None value again; callback should not be called
    p.set(None)
    assert callback_calls == []

    # Simulate a get that changes the value (from None to 30)
    get_cmd_val = 30
    p.get()
    assert callback_calls == [(None, 30, 'get')]

    # Reset callback_calls
    callback_calls.clear()

    # Simulate a get that changes the value to None (from 30 to None)
    get_cmd_val = None    
    p.get()
    assert callback_calls == [(30, None, 'get')]

def test_parameter_on_value_change_update_callback() -> None:
    callback_calls_initial = []
    callback_calls_new = []

    def initial_callback(old_value, new_value, source):
        callback_calls_initial.append((old_value, new_value, source))

    def new_callback(old_value, new_value, source):
        callback_calls_new.append((old_value, new_value, source))

    # Create a parameter with the initial on_value_change callback
    p = OnValueChangeParameter(
        name="p",
        set_cmd=None,
        get_cmd=None,
        vals=vals.Anything(), 
        on_value_change=initial_callback,
    )

    # Set a new value to trigger the initial callback
    p.set(10)
    assert callback_calls_initial == [(None, 10, 'set')]
    assert callback_calls_new == []

    # Update the callback to a new one
    p.on_value_change = new_callback

    # Reset initial callback calls
    callback_calls_initial.clear()

    # Set another new value to trigger the new callback
    p.set(20)
    assert callback_calls_initial == []
    assert callback_calls_new == [(10, 20, 'set')]

def test_parameter_on_value_change_invalid_assignment() -> None:
    """
    Test that setting on_value_change to a non-callable and non-None value raises a TypeError.
    """
    # Create a parameter without an initial callback
    p = OnValueChangeParameter(
        name="p",
        set_cmd=None,
        get_cmd=None,
        vals=vals.Anything(),
    )

    # Attempt to set on_value_change to an integer
    with pytest.raises(TypeError, match="on_value_change must be a callable or None"):
        p.on_value_change = 42  # type: ignore[assignment]  # Invalid: not callable and not None

    # Attempt to set on_value_change to a string
    with pytest.raises(TypeError, match="on_value_change must be a callable or None"):
        p.on_value_change = "not a function"  # type: ignore[assignment]  # Invalid: not callable and not None

    # Attempt to set on_value_change to a list
    with pytest.raises(TypeError, match="on_value_change must be a callable or None"):
        p.on_value_change = [1, 2, 3]  # type: ignore[assignment]  # Invalid: not callable and not None

    # Ensure that setting to a callable works without raising an error
    def valid_callback(old_value, new_value, source):
        pass

    try:
        p.on_value_change = valid_callback  # Valid: callable
    except TypeError:
        pytest.fail("Setting on_value_change to a callable should not raise TypeError")

    # Ensure that setting to None works without raising an error
    try:
        p.on_value_change = None  # Valid: None
    except TypeError:
        pytest.fail("Setting on_value_change to None should not raise TypeError")
