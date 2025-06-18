import pytest
from typing import Any, List, Tuple, Dict
from qcodes.instrument import Instrument
from qcodes.parameters import Parameter, SetCacheValueOnResetParameterMixin
from qcodes.utils import QCoDeSDeprecationWarning

class MockResetInstrument(Instrument):
    """
    A mock instrument that records calls to perform_reset.
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.reset_call_args: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

    def perform_reset(self, *args: Any, **kwargs: Any):
        self.reset_call_args.append((args, kwargs))
        SetCacheValueOnResetParameterMixin.reset_group("reset_group_general")


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
    return instr


@pytest.fixture(autouse=True)
def clear_reset_registry():
    SetCacheValueOnResetParameterMixin._reset_group_registry.clear()


def test_cache_resets_to_value(store, reset_instr) -> None:
    test_param: ResetTestParameter = reset_instr.add_parameter(
        name="test_param",
        parameter_class=ResetTestParameter,
        reset_group_names=["reset_group_general"],
        cache_value_after_reset=42,
        set_cmd=lambda x: store.update({"reset_param": x}),
        docstring="A parameter that resets its cache to 42."
    )

    test_param.set(10)
    assert test_param.get() == 10
    reset_instr.perform_reset()
    assert test_param.get() == 42


def test_warning_if_cache_value_unset(store, reset_instr) -> None:
    with pytest.warns(UserWarning, match="cache_value_after_reset for parameter 'test_param' is not set"):
        reset_instr.add_parameter(
            name="test_param",
            parameter_class=ResetTestParameter,
            reset_group_names=["reset_group_general"],
            set_cmd=lambda x: store.update({"reset_param_unset": x}),
            docstring="A parameter with no reset value set."
        )


def test_cache_resets_to_none(store, reset_instr) -> None:
    test_param: ResetTestParameter = reset_instr.add_parameter(
        name="test_param_none",
        parameter_class=ResetTestParameter,
        reset_group_names=["reset_group_general"],
        cache_value_after_reset=None,
        set_cmd=lambda x: store.update({"reset_param_none": x}),
        docstring="A parameter resetting its cache to None."
    )

    test_param.set(25)
    assert test_param.get() == 25
    reset_instr.perform_reset()
    assert test_param.get() is None


def test_set_parser_with_default_get_parser(store, reset_instr) -> None:
    test_param: ResetTestParameter = reset_instr.add_parameter(
        name="test_param",
        parameter_class=ResetTestParameter,
        reset_group_names=["reset_group_general"],
        cache_value_after_reset=10,
        set_cmd=lambda x: store.update({"reset_param_parsers": x}),
        set_parser=lambda v: v * 3,
        docstring="A parameter with set_parser and default get_parser."
    )

    test_param.set(5)
    assert test_param.get() == 5
    reset_instr.perform_reset()
    assert test_param.get() == 10


def test_direct_cache_update_and_reset(store, reset_instr) -> None:
    test_param: ResetTestParameter = reset_instr.add_parameter(
        name="test_param",
        parameter_class=ResetTestParameter,
        reset_group_names=["reset_group_general"],
        cache_value_after_reset=50,
        set_cmd=lambda x: store.update({"reset_param_direct": x}),
        docstring="A parameter testing direct cache update and reset."
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
                    reset_group_names=["reset_group_general"],
                    cache_value_after_reset=42,
                    set_cmd=lambda x: None,
                    get_cmd=lambda: 100,
                    docstring="A parameter incorrectly supplying get_cmd."
                )


def test_error_if_get_parser_supplied(reset_instr) -> None:
    with pytest.warns(QCoDeSDeprecationWarning, match="does not correctly pass kwargs"):
        # with pytest.raises(KeyError, match="Duplicate parameter name managed_param on instrument"):
            with pytest.raises(TypeError, match="Supplying 'get_parser' is not allowed"):
                reset_instr.add_parameter(
                    name="test_param_get_parser_error",
                    parameter_class=ResetTestParameter,
                    reset_group_names=["reset_group_general"],
                    cache_value_after_reset=42,
                    set_cmd=lambda x: None,
                    get_parser=lambda x: x + 1,
                    docstring="A parameter incorrectly supplying get_parser."
                )


def test_warning_no_callbacks_for_group() -> None:
    with pytest.warns(UserWarning, match="No callbacks registered for reset group 'empty_group'"):
        SetCacheValueOnResetParameterMixin.reset_group("empty_group")


def test_multiple_callbacks_in_group(store, reset_instr) -> None:
    call_order = []

    def callback_one():
        call_order.append("callback_one")
        store.update({"callback_one": True})

    def callback_two():
        call_order.append("callback_two")
        store.update({"callback_two": True})

    SetCacheValueOnResetParameterMixin.register_reset_callback("multi_callback_group", callback_one)
    SetCacheValueOnResetParameterMixin.register_reset_callback("multi_callback_group", callback_two)

    SetCacheValueOnResetParameterMixin.reset_group("multi_callback_group")
    assert call_order == ["callback_one", "callback_two"]
    assert store["callback_one"] is True
    assert store["callback_two"] is True


def test_parameter_in_multiple_reset_groups(store, reset_instr) -> None:
    test_param: ResetTestParameter = reset_instr.add_parameter(
        name="test_param",
        parameter_class=ResetTestParameter,
        reset_group_names=["reset_group_general", "group_b"],
        cache_value_after_reset=100,
        set_cmd=lambda x: store.update({"multi_group_param": x}),
        docstring="A parameter in multiple reset groups."
    )

    test_param.set(50)
    reset_instr.perform_reset()
    assert test_param.get() == 100

    test_param.set(75)
    SetCacheValueOnResetParameterMixin.reset_group("group_b")
    assert test_param.get() == 100


def test_callback_execution_order(reset_instr) -> None:
    execution_sequence = []

    def first_callback():
        execution_sequence.append("first")

    def second_callback():
        execution_sequence.append("second")

    SetCacheValueOnResetParameterMixin.register_reset_callback("order_group", first_callback)
    SetCacheValueOnResetParameterMixin.register_reset_callback("order_group", second_callback)

    SetCacheValueOnResetParameterMixin.reset_group("order_group")
    assert execution_sequence == ["first", "second"]
