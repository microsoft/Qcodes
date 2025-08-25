from typing import Any, cast

import pytest

from qcodes.extensions.parameters import OnCacheChangeParameterMixin
from qcodes.instrument import Instrument
from qcodes.parameters import Parameter


class MockInstrument(Instrument):
    """
    A mock instrument to host parameters.
    """

    def __init__(self, name: str):
        super().__init__(name)


class OnCacheChangeParameter(OnCacheChangeParameterMixin, Parameter):
    """
    A parameter invoking callbacks on cache changes.
    """

    pass


@pytest.fixture
def store():
    """
    Provides a dictionary to store parameter values.
    """
    return {}


@pytest.fixture
def callback_flag():
    """
    Provides a mutable flag to track callback invocation.
    """
    return {"called": False}


@pytest.fixture
def callback(callback_flag):
    """
    Provides a callback that sets callback_flag["called"] to True.
    """

    def _callback(**kwargs: Any) -> None:
        callback_flag["called"] = True

    return _callback


@pytest.fixture
def callback_data():
    """
    Provides a list to store callback invocation data.
    """
    return []


@pytest.fixture
def data_callback(callback_data):
    """
    Provides a callback that records (args, kwargs) into callback_data.
    """

    def _callback(**kwargs):
        callback_data.append(kwargs)

    return _callback


@pytest.fixture
def mock_instr():
    """
    Provides a mock instrument to host parameters.
    """
    instr = MockInstrument("mock_instr")
    yield instr
    instr.close()


@pytest.mark.parametrize("invalid_callback", ["not_a_callable", 123, [], {}])
def test_error_on_non_callable_callback(mock_instr, invalid_callback) -> None:
    test_parameter: OnCacheChangeParameter = mock_instr.add_parameter(
        name="test_parameter",
        parameter_class=OnCacheChangeParameter,
        docstring="A parameter for testing non-callable callbacks.",
    )
    with pytest.raises(TypeError):
        test_parameter.on_cache_change = invalid_callback


def test_no_callback_invocation_on_init_or_get(
    store, callback_flag, callback, mock_instr
) -> None:
    test_parameter: OnCacheChangeParameter = mock_instr.add_parameter(
        name="test_parameter",
        parameter_class=OnCacheChangeParameter,
        on_cache_change=callback,
        set_cmd=lambda x: store.update({"value": x}),
        get_cmd=lambda: store.get("value"),
        initial_value=12,
        docstring="A parameter with initial value set to 12.",
    )
    assert test_parameter.get() == 12
    assert not callback_flag["called"], "Callback invoked unexpectedly."


def test_callback_invoked_on_set(store, callback_flag, callback, mock_instr) -> None:
    test_parameter: OnCacheChangeParameter = mock_instr.add_parameter(
        name="test_parameter",
        parameter_class=OnCacheChangeParameter,
        on_cache_change=callback,
        set_cmd=lambda x: store.update({"value": x}),
        get_cmd=lambda: store.get("value"),
        docstring="A parameter invoking callback on set.",
    )
    test_parameter.set(42)
    assert callback_flag["called"], "Callback not invoked on cache change."


def test_changing_callback_after_init(
    store, callback_flag, callback, mock_instr
) -> None:
    test_parameter: OnCacheChangeParameter = mock_instr.add_parameter(
        name="test_parameter",
        parameter_class=OnCacheChangeParameter,
        set_cmd=lambda x: store.update({"value": x}),
        get_cmd=lambda: store.get("value"),
        docstring="A parameter without initial callback.",
    )
    test_parameter.set(10)
    assert not callback_flag["called"]

    test_parameter.on_cache_change = callback
    test_parameter.set(20)
    assert callback_flag["called"]


def test_callback_on_get_value_change(callback_flag, callback, mock_instr) -> None:
    get_reply = None

    test_parameter: OnCacheChangeParameter = mock_instr.add_parameter(
        name="test_parameter",
        parameter_class=OnCacheChangeParameter,
        get_cmd=lambda: get_reply,
        on_cache_change=callback,
        docstring="A parameter with dynamic get behavior.",
    )

    assert test_parameter.get() is None

    get_reply = 42
    test_parameter.get()
    assert callback_flag["called"], "Callback not invoked on value change."


def test_callback_on_direct_cache_update(
    store, callback_data, data_callback, mock_instr
) -> None:
    test_parameter: OnCacheChangeParameter = mock_instr.add_parameter(
        name="test_parameter",
        parameter_class=OnCacheChangeParameter,
        on_cache_change=data_callback,
        set_cmd=lambda x: store.update({"value": x}),
        get_cmd=lambda: store.get("value"),
        docstring="A parameter testing direct cache update.",
    )
    test_parameter.cache._update_with(value=5, raw_value=5)
    assert len(callback_data) == 1
    assert callback_data[0]["value_old"] is None
    assert callback_data[0]["value_new"] == 5


def test_no_callback_if_value_unchanged(
    store, callback_data, data_callback, mock_instr
) -> None:
    test_parameter: OnCacheChangeParameter = mock_instr.add_parameter(
        name="test_parameter",
        parameter_class=OnCacheChangeParameter,
        on_cache_change=data_callback,
        set_cmd=lambda x: store.update({"value": x}),
        get_cmd=lambda: store.get("value"),
        docstring="A parameter testing unchanged values.",
    )

    test_parameter.set(10)
    assert len(callback_data) == 1
    callback_data.clear()

    test_parameter.set(10)
    assert not callback_data


def test_callback_value_parsers(store, callback_data, mock_instr) -> None:
    def cb(**kwargs):
        callback_data.append(
            (
                kwargs.get("value_old"),
                kwargs.get("value_new"),
                kwargs.get("raw_value_old"),
                kwargs.get("raw_value_new"),
            )
        )

    mock_instr.test_parameter = cast(
        "OnCacheChangeParameter",
        mock_instr.add_parameter(
            name="test_parameter",
            parameter_class=OnCacheChangeParameter,
            on_cache_change=cb,
            set_cmd=lambda x: store.update({"value": x}),
            get_cmd=lambda: None if store.get("value") is None else store.get("value"),
            set_parser=lambda v: v * 2,
            get_parser=lambda v: None if v is None else v / 2,
            docstring="A parameter with set_parser.",
        ),
    )

    assert mock_instr.test_parameter.get() is None

    mock_instr.test_parameter.set(5)
    assert callback_data == [(None, 5, None, 10)]
    callback_data.clear()
    assert mock_instr.test_parameter.get() == 5
    assert callback_data == []
    callback_data.clear()

    mock_instr.test_parameter.set(7)
    assert callback_data == [(5, 7, 10, 14)]
    callback_data.clear()
    assert mock_instr.test_parameter.get() == 7
    assert callback_data == []
    callback_data.clear()

    mock_instr.test_parameter.set(7)
    assert not callback_data

    mock_instr.test_parameter.cache._update_with(value=110, raw_value=100)
    assert callback_data == [(7, 110, 14, 100)]
