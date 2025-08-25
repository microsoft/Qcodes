from typing import Any

import pytest

from qcodes.extensions.parameters import SetCacheValueOnResetParameterMixin
from qcodes.instrument import Instrument
from qcodes.parameters import Parameter
from qcodes.utils import QCoDeSDeprecationWarning


class MockResetInstrument(Instrument):
    """
    A mock instrument that records calls to perform_reset.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.reset_call_args: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def perform_reset(self, *args: Any, **kwargs: Any):
        self.reset_call_args.append((args, kwargs))
        SetCacheValueOnResetParameterMixin.trigger_group("reset_group_general")


class ResetTestParameter(SetCacheValueOnResetParameterMixin, Parameter):
    """
    A parameter resetting its cache value on instrument reset.
    """

    pass


@pytest.fixture
def store():
    """
    Provides a dictionary to store parameter values.
    """
    return {}


@pytest.fixture
def reset_instr():
    """
    Provides a mock instrument with reset capability.
    """
    instr = MockResetInstrument("mock_reset_instr")
    yield instr
    instr.close()


@pytest.fixture(autouse=True)
def clear_reset_registry():
    SetCacheValueOnResetParameterMixin._group_registry.clear()


def test_cache_resets_to_value(store, reset_instr) -> None:
    test_param: ResetTestParameter = reset_instr.add_parameter(
        name="test_param",
        parameter_class=ResetTestParameter,
        group_names=["reset_group_general"],
        cache_value_after_reset=42,
        set_cmd=lambda x: store.update({"reset_param": x}),
        docstring="A parameter that resets its cache to 42.",
    )

    test_param.set(10)
    assert test_param.get() == 10
    reset_instr.perform_reset()
    assert test_param.get() == 42


def test_warning_if_cache_value_unset(store, reset_instr) -> None:
    with pytest.warns(
        UserWarning,
        match="cache_value_after_reset for parameter 'test_param' is not set",
    ):
        reset_instr.add_parameter(
            name="test_param",
            parameter_class=ResetTestParameter,
            group_names=["reset_group_general"],
            set_cmd=lambda x: store.update({"reset_param_unset": x}),
            docstring="A parameter with no reset value set.",
        )


def test_cache_resets_to_none(store, reset_instr) -> None:
    test_param: ResetTestParameter = reset_instr.add_parameter(
        name="test_param_none",
        parameter_class=ResetTestParameter,
        group_names=["reset_group_general"],
        cache_value_after_reset=None,
        set_cmd=lambda x: store.update({"reset_param_none": x}),
        docstring="A parameter resetting its cache to None.",
    )

    test_param.set(25)
    assert test_param.get() == 25
    reset_instr.perform_reset()
    assert test_param.get() is None


def test_set_parser_with_default_get_parser(store, reset_instr) -> None:
    test_param: ResetTestParameter = reset_instr.add_parameter(
        name="test_param",
        parameter_class=ResetTestParameter,
        group_names=["reset_group_general"],
        cache_value_after_reset=10,
        set_cmd=lambda x: store.update({"reset_param_parsers": x}),
        set_parser=lambda v: v * 3,
        docstring="A parameter with set_parser and default get_parser.",
    )

    test_param.set(5)
    assert test_param.get() == 5
    assert test_param.get_raw() == 15
    reset_instr.perform_reset()
    assert test_param.get() == 10
    assert test_param.get_raw() == 30


def test_direct_cache_update_and_reset(store, reset_instr) -> None:
    test_param: ResetTestParameter = reset_instr.add_parameter(
        name="test_param",
        parameter_class=ResetTestParameter,
        group_names=["reset_group_general"],
        cache_value_after_reset=50,
        set_cmd=lambda x: store.update({"reset_param_direct": x}),
        docstring="A parameter testing direct cache update and reset.",
    )

    test_param.set(30)
    test_param.cache.set(35)
    assert test_param.get() == 35
    reset_instr.perform_reset()
    assert test_param.get() == 50


def test_error_if_get_cmd_supplied(reset_instr) -> None:
    with pytest.warns(QCoDeSDeprecationWarning, match="does not correctly pass kwargs"):
        # with pytest.raises(KeyError, match="Duplicate parameter name managed_param on instrument"):
        with pytest.raises(TypeError, match="without 'get_cmd'"):
            reset_instr.add_parameter(
                name="test_param_error",
                parameter_class=ResetTestParameter,
                group_names=["reset_group_general"],
                cache_value_after_reset=42,
                set_cmd=lambda x: None,
                get_cmd=lambda: 100,
                docstring="A parameter incorrectly supplying get_cmd.",
            )


def test_error_if_get_parser_supplied(reset_instr) -> None:
    with pytest.warns(QCoDeSDeprecationWarning, match="does not correctly pass kwargs"):
        # with pytest.raises(KeyError, match="Duplicate parameter name managed_param on instrument"):
        with pytest.raises(TypeError, match="Supplying 'get_parser' is not allowed"):
            reset_instr.add_parameter(
                name="test_param_get_parser_error",
                parameter_class=ResetTestParameter,
                group_names=["reset_group_general"],
                cache_value_after_reset=42,
                set_cmd=lambda x: None,
                get_parser=lambda x: x + 1,
                docstring="A parameter incorrectly supplying get_parser.",
            )


def test_parameter_in_multiple_reset_groups(store, reset_instr) -> None:
    test_param: ResetTestParameter = reset_instr.add_parameter(
        name="test_param",
        parameter_class=ResetTestParameter,
        group_names=["reset_group_general", "group_b"],
        cache_value_after_reset=100,
        set_cmd=lambda x: store.update({"multi_group_param": x}),
        docstring="A parameter in multiple reset groups.",
    )

    test_param.set(50)
    reset_instr.perform_reset()
    assert test_param.get() == 100

    test_param.set(75)
    SetCacheValueOnResetParameterMixin.trigger_group("group_b")
    assert test_param.get() == 100


def test_get_raw_with_val_mapping(reset_instr) -> None:
    class MyParam(SetCacheValueOnResetParameterMixin, Parameter):
        pass

    p = reset_instr.add_parameter(
        name="valmap_param",
        parameter_class=MyParam,
        group_names=["reset_group_general"],
        cache_value_after_reset="ASCII",
        val_mapping={"ASCII": "1", "Binary": "2"},
        set_cmd=lambda x: None,
    )

    p.set("Binary")
    assert p.get() == "Binary"
    assert p.get_raw() == "2"
    reset_instr.perform_reset()
    assert p.get() == "ASCII"
    assert p.get_raw() == "1"


def test_warning_if_group_names_is_none(store, reset_instr):
    with pytest.warns(UserWarning, match="No group_name"):
        reset_instr.add_parameter(
            name="test_param",
            parameter_class=ResetTestParameter,
            cache_value_after_reset=42,
            group_names=None,
            set_cmd=lambda x: store.update({"reset_param": x}),
        )


def test_warning_if_group_names_missing(store, reset_instr):
    with pytest.warns(UserWarning, match="No group_name"):
        reset_instr.add_parameter(
            name="test_param",
            parameter_class=ResetTestParameter,
            cache_value_after_reset=42,
            set_cmd=lambda x: store.update({"reset_param": x}),
        )


def test_typeerror_if_group_names_invalid(store, reset_instr):
    with pytest.warns(QCoDeSDeprecationWarning):
        with pytest.raises(
            TypeError, match="group_names must be a list of strings or None"
        ):
            reset_instr.add_parameter(
                name="test_param",
                parameter_class=ResetTestParameter,
                cache_value_after_reset=42,
                group_names=123,
                set_cmd=lambda x: store.update({"reset_param": x}),
            )
