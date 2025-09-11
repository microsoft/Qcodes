import re
from typing import Any, cast

import pytest

from qcodes.extensions.parameters import InterdependentParameterMixin
from qcodes.instrument import Instrument
from qcodes.parameters import Parameter
from qcodes.utils import QCoDeSDeprecationWarning


class MockInstrument(Instrument):
    """
    A mock instrument that can host parameters.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)


class InterdependentParameter(InterdependentParameterMixin, Parameter):
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

    def _callback(*args, **kwargs):
        callback_data.append((args, kwargs))

    return _callback


@pytest.fixture
def mock_instr():
    """
    Provides a mock instrument for interdependent parameter tests.
    """
    instr = MockInstrument("mock_instr")
    yield instr
    instr.close()


def test_dependency_update_invoked_on_change(
    store, callback_flag, callback, mock_instr
) -> None:
    mock_instr.some_param = cast(
        "InterdependentParameter",
        mock_instr.add_parameter(
            name="some_param",
            parameter_class=InterdependentParameter,
            set_cmd=lambda x: store.update({"some": x}),
            get_cmd=lambda: store.get("some"),
            docstring="Parameter some_param represents a primary parameter.",
        ),
    )
    """Parameter some_param represents a primary parameter."""

    mock_instr.managed_param = cast(
        "InterdependentParameter",
        mock_instr.add_parameter(
            name="managed_param",
            parameter_class=InterdependentParameter,
            dependent_on=["some_param"],
            dependency_update_method=callback,
            set_cmd=lambda x: store.update({"managed": x}),
            get_cmd=lambda: store.get("managed"),
            docstring="Parameter managed_param depends on some_param.",
        ),
    )
    """Parameter managed_param depends on some_param."""

    mock_instr.some_param.set(42)
    assert callback_flag["called"], "dependency_update_method was not called."


def test_adding_dependent_parameter_later(
    store, callback_flag, callback, mock_instr
) -> None:
    mock_instr.some_param = cast(
        "InterdependentParameter",
        mock_instr.add_parameter(
            name="some_param",
            parameter_class=InterdependentParameter,
            set_cmd=lambda x: store.update({"some": x}),
            get_cmd=lambda: store.get("some"),
            docstring="Parameter some_param represents a primary parameter.",
        ),
    )
    """Parameter some_param represents a primary parameter."""

    mock_instr.managed_param = cast(
        "InterdependentParameter",
        mock_instr.add_parameter(
            name="managed_param",
            parameter_class=InterdependentParameter,
            dependent_on=["some_param"],
            dependency_update_method=callback,
            set_cmd=lambda x: store.update({"managed": x}),
            get_cmd=lambda: store.get("managed"),
            docstring="Parameter managed_param depends on some_param.",
        ),
    )
    """Parameter managed_param depends on some_param."""

    mock_instr.some_param.add_dependent_parameter(mock_instr.managed_param)
    mock_instr.some_param.set(100)
    assert callback_flag["called"], "dependency_update_method was not called."


def test_error_on_non_interdependent_dependency(store, mock_instr) -> None:
    mock_instr.not_interdependent_param = cast(
        "InterdependentParameter",
        mock_instr.add_parameter(
            name="not_interdependent_param",
            set_cmd=lambda x: store.update({"not_interdep": x}),
            get_cmd=lambda: store.get("not_interdep"),
            docstring="A non-interdependent parameter.",
        ),
    )
    """A non-interdependent parameter."""

    with pytest.warns(QCoDeSDeprecationWarning, match="does not correctly pass kwargs"):
        with pytest.raises(
            KeyError, match="Duplicate parameter name managed_param on instrument"
        ):
            with pytest.raises(
                TypeError, match="must be an instance of InterdependentParameterMixin"
            ):
                mock_instr.managed_param = cast(
                    "InterdependentParameter",
                    mock_instr.add_parameter(
                        name="managed_param",
                        parameter_class=InterdependentParameter,
                        dependent_on=["not_interdependent_param"],
                        set_cmd=lambda x: store.update({"managed": x}),
                        get_cmd=lambda: store.get("managed"),
                        docstring="Parameter managed_param depends on a non-interdependent param.",
                    ),
                )
                """Parameter managed_param depends on a non-interdependent param."""


def test_parsers_and_dependency_propagation(store, mock_instr) -> None:
    def dependency_update():
        mock_instr.managed_param.set(999)

    mock_instr.some_param = cast(
        "InterdependentParameter",
        mock_instr.add_parameter(
            name="some_param",
            parameter_class=InterdependentParameter,
            set_cmd=lambda x: store.update({"some": x}),
            get_cmd=lambda: store.get("some"),
            set_parser=lambda v: v * 2,
            get_parser=lambda v: v + 1 if v is not None else None,
            docstring="Parameter some_param with parsers.",
        ),
    )
    """Parameter some_param with parsers."""

    mock_instr.managed_param = cast(
        "InterdependentParameter",
        mock_instr.add_parameter(
            name="managed_param",
            parameter_class=InterdependentParameter,
            dependent_on=["some_param"],
            dependency_update_method=dependency_update,
            set_cmd=lambda x: store.update({"managed": x}),
            get_cmd=lambda: store.get("managed"),
            docstring="Parameter managed_param depends on some_param.",
        ),
    )
    """Parameter managed_param depends on some_param."""

    assert mock_instr.some_param.get() is None
    mock_instr.some_param.set(10)
    assert mock_instr.some_param.get() == 21, "Parser not applied correctly."
    assert mock_instr.managed_param.get() == 999, "Dependency not updated."


def test_typeerror_if_dependency_update_method_invalid(mock_instr):
    param = mock_instr.add_parameter(
        name="some_param",
        parameter_class=InterdependentParameter,
        set_cmd=lambda x: None,
        get_cmd=lambda: None,
    )
    # Assign an invalid value (not callable, not None)
    with pytest.raises(
        TypeError, match=re.escape("dependency_update_method must be callable or None.")
    ):
        param.dependency_update_method = 123


def test_typeerror_if_dependent_on_invalid(mock_instr):
    param = mock_instr.add_parameter(
        name="some_param",
        parameter_class=InterdependentParameter,
        set_cmd=lambda x: None,
        get_cmd=lambda: None,
    )
    # Assign a non-list
    with pytest.raises(
        TypeError, match=re.escape("dependent_on must be a list of strings.")
    ):
        param.dependent_on = "not_a_list"

    # Assign a list with non-string
    with pytest.raises(
        TypeError, match=re.escape("dependent_on must be a list of strings.")
    ):
        param.dependent_on = [123, None]
