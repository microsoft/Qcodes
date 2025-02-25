from __future__ import annotations

import pytest

from qcodes import validators
from qcodes.parameters import Parameter


# Basic functionality
def test_set_value_changed_callback():
    """Test setting and removing a value changed callback"""
    called_with = []

    def callback(param, value):
        called_with.append((param, value))

    p = Parameter("p", set_cmd=None, get_cmd=None)

    p.set_value_changed_callback(callback)
    p.set(1)
    assert len(called_with) == 1
    assert called_with[0][0] == p
    assert called_with[0][1] == 1

    p.set_value_changed_callback(None)
    p.set(2)
    assert len(called_with) == 1


def test_callback_initial_value():
    """Test callback behavior with initial value"""
    called_values = []

    def callback(param, value):
        called_values.append(value)

    p = Parameter("p", set_cmd=None, get_cmd=None, initial_value=5)
    p.set_value_changed_callback(callback)

    assert called_values == []
    p.set(6)
    assert called_values == [6]


# Multiple callbacks
def test_multiple_set_callbacks():
    """Test callback is called each time set is called"""
    call_count = 0

    def callback(param, value):
        nonlocal call_count
        call_count += 1

    p = Parameter("p", set_cmd=None, get_cmd=None)
    p.set_value_changed_callback(callback)

    p.set(1)
    p.set(1)
    p.set(2)

    assert call_count == 3


def test_multiple_callbacks_replace():
    """Test that setting a new callback replaces the old one"""
    calls_a = []
    calls_b = []

    def callback_a(param, value):
        calls_a.append(value)

    def callback_b(param, value):
        calls_b.append(value)

    p = Parameter("p", set_cmd=None, get_cmd=None)
    p.set_value_changed_callback(callback_a)
    p.set(1)
    p.set_value_changed_callback(callback_b)
    p.set(2)

    assert calls_a == [1]
    assert calls_b == [2]


def test_callback_lifecycle():
    """Test complete callback lifecycle: add, call, remove"""
    calls_a = []
    calls_b = []

    def callback_a(param, value):
        calls_a.append(value)

    def callback_b(param, value):
        calls_b.append(value)

    p = Parameter("p", set_cmd=None, get_cmd=None)

    p.set_value_changed_callback(callback_a)
    p.set_value_changed_callback(callback_b)
    p.set(1)
    assert calls_a == [] and calls_b == [1]

    p.set_value_changed_callback(None)
    p.set(2)
    assert calls_a == [] and calls_b == [1]


# Edge cases
def test_callback_with_none_value():
    """Test callback behavior when setting None value"""
    called_values = []

    def callback(param, value):
        called_values.append(value)

    p = Parameter("p", set_cmd=None, get_cmd=None)
    p.set_value_changed_callback(callback)
    p.set(None)
    assert called_values == [None]


def test_remove_nonexistent_callback():
    """Test removing a callback that was never set"""
    p = Parameter("p", set_cmd=None, get_cmd=None)
    p.set_value_changed_callback(None)
    p.set(1)


# Validation
def test_callback_with_validation():
    """Test callback is called only after validation passes"""
    called_values = []

    def callback(param, value):
        called_values.append(value)

    p = Parameter("p", set_cmd=None, get_cmd=None, vals=validators.Numbers(0, 10))
    p.set_value_changed_callback(callback)

    with pytest.raises(ValueError):
        p.set(20)

    assert len(called_values) == 0


def test_invalid_callback():
    """Test setting an invalid callback raises TypeError"""
    p = Parameter("p", set_cmd=None, get_cmd=None)

    with pytest.raises(TypeError, match="Callback must be type callable or None"):
        p.set_value_changed_callback("not_a_callback")


def test_callback_exception_handling(caplog):
    """Test that exceptions in callbacks are caught and logged"""

    def failing_callback(param, value):
        raise ValueError("Callback failed")

    p = Parameter("p", set_cmd=None, get_cmd=None)
    p.set_value_changed_callback(failing_callback)

    p.set(1)

    assert "Exception while running parameter callback" in caplog.text


# Advanced features
def test_callback_with_cache():
    """Test callback behavior with cache operations"""
    called_values = []

    def callback(param, value):
        called_values.append(value)

    p = Parameter("p", set_cmd=None, get_cmd=None)
    p.set_value_changed_callback(callback)

    p.cache.set(1)
    assert called_values == []

    p.set(2)
    assert called_values == [2]


def test_callback_thread_safety():
    """Test callback behavior with rapid value changes"""
    import threading
    import time

    called_values = []

    def callback(param, value):
        time.sleep(0.01)
        called_values.append(value)

    p = Parameter("p", set_cmd=None, get_cmd=None)
    p.set_value_changed_callback(callback)

    def set_value():
        p.set(1)
        p.set(2)

    t1 = threading.Thread(target=set_value)
    t2 = threading.Thread(target=set_value)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(called_values) == 4
