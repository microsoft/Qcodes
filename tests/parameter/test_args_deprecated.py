"""Tests that passing positional arguments (beyond ``name``) to
ParameterBase, Parameter, and DelegateParameter triggers a
QCoDeSDeprecationWarning.
"""

from __future__ import annotations

from typing import Any

import pytest

from qcodes.instrument import Instrument
from qcodes.parameters import (
    ArrayParameter,
    DelegateParameter,
    GroupParameter,
    Parameter,
    ParameterBase,
)
from qcodes.parameters.grouped_parameter import (
    DelegateGroup,
    DelegateGroupParameter,
    GroupedParameter,
)
from qcodes.utils import QCoDeSDeprecationWarning


class _MockInstrument(Instrument):
    """A minimal instrument for testing."""

    def __init__(self, name: str = "mock"):
        super().__init__(name)


@pytest.fixture
def mock_instrument() -> Any:
    inst = _MockInstrument("dup_test")
    yield inst
    inst.close()


# Minimal concrete subclass of ParameterBase for testing
class _ConcreteParameterBase(ParameterBase):
    def get_raw(self) -> Any:
        return 0


class TestParameterBasePositionalArgs:
    """ParameterBase should warn when arguments after ``name`` are positional."""

    def test_single_positional_arg_warns(self) -> None:
        with pytest.warns(
            QCoDeSDeprecationWarning,
            match="Passing 'instrument' as positional argument",
        ):
            _ConcreteParameterBase("test", None)

    def test_multiple_positional_args_warn(self) -> None:
        with pytest.warns(
            QCoDeSDeprecationWarning,
            match="'instrument', 'snapshot_get'",
        ):
            _ConcreteParameterBase("test", None, True)

    def test_keyword_args_do_not_warn(self) -> None:
        # No warning should be raised when all args are keyword-only
        p = _ConcreteParameterBase("test", instrument=None, snapshot_get=True)
        assert p.name == "test"

    def test_duplicate_positional_and_keyword_raises(
        self, mock_instrument: _MockInstrument
    ) -> None:
        with pytest.raises(
            TypeError,
            match="got multiple values for argument 'instrument'",
        ):
            _ConcreteParameterBase("test", None, instrument=mock_instrument)

    def test_too_many_positional_args_raises(self) -> None:
        # More positional args than defined parameter names
        too_many = (None,) * 25
        with pytest.raises(TypeError, match="takes at most"):
            _ConcreteParameterBase("test", *too_many)


class TestParameterPositionalArgs:
    """Parameter should warn when arguments after ``name`` are positional."""

    def test_single_positional_arg_warns(self) -> None:
        with pytest.warns(
            QCoDeSDeprecationWarning,
            match="Passing 'instrument' as positional argument",
        ):
            Parameter("test", None, set_cmd=None)

    def test_multiple_positional_args_warn(self) -> None:
        with pytest.warns(
            QCoDeSDeprecationWarning,
            match="'instrument', 'label'",
        ):
            Parameter("test", None, "my label", set_cmd=None)

    def test_keyword_args_do_not_warn(self) -> None:
        p = Parameter("test", instrument=None, label="my label", set_cmd=None)
        assert p.name == "test"
        assert p.label == "my label"

    def test_duplicate_positional_and_keyword_raises(
        self, mock_instrument: _MockInstrument
    ) -> None:
        with pytest.raises(
            TypeError,
            match="got multiple values for argument 'instrument'",
        ):
            Parameter("test", None, instrument=mock_instrument)

    def test_too_many_positional_args_raises(self) -> None:
        too_many = (None,) * 15
        with pytest.raises(TypeError, match="takes at most"):
            Parameter("test", *too_many)


class TestDelegateParameterPositionalArgs:
    """DelegateParameter should warn when extra positional args are passed."""

    def test_positional_args_warn(self) -> None:
        source = Parameter("source", set_cmd=None)
        with pytest.warns(QCoDeSDeprecationWarning):
            DelegateParameter("test", source, None)

    def test_keyword_args_do_not_warn(self) -> None:
        source = Parameter("source", set_cmd=None)
        p = DelegateParameter("test", source, instrument=None)
        assert p.name == "test"


# Minimal concrete subclass of ArrayParameter for testing
class _ConcreteArrayParameter(ArrayParameter):
    def get_raw(self) -> Any:
        return [0]


class TestArrayParameterPositionalArgs:
    """ArrayParameter should warn when arguments after ``name`` are positional."""

    def test_single_positional_arg_warns(self) -> None:
        with pytest.warns(
            QCoDeSDeprecationWarning,
            match="Passing 'shape' as positional argument",
        ):
            _ConcreteArrayParameter("test", (3,))

    def test_multiple_positional_args_warn(self) -> None:
        with pytest.warns(
            QCoDeSDeprecationWarning,
            match="'shape', 'instrument'",
        ):
            _ConcreteArrayParameter("test", (3,), None)

    def test_keyword_args_do_not_warn(self) -> None:
        p = _ConcreteArrayParameter("test", shape=(3,), instrument=None)
        assert p.name == "test"
        assert p.shape == (3,)

    def test_duplicate_positional_and_keyword_raises(self) -> None:
        with pytest.raises(
            TypeError,
            match="got multiple values for argument 'shape'",
        ):
            _ConcreteArrayParameter("test", (3,), shape=(3,))

    def test_too_many_positional_args_raises(self) -> None:
        too_many = (None,) * 20
        with pytest.raises(TypeError, match="takes at most"):
            _ConcreteArrayParameter("test", *too_many)

    def test_missing_shape_raises(self) -> None:
        with pytest.raises(
            TypeError,
            match="missing required keyword argument: 'shape'",
        ):
            _ConcreteArrayParameter("test")


class TestGroupParameterPositionalArgs:
    """GroupParameter should warn when arguments after ``name`` are positional."""

    def test_single_positional_arg_warns(self) -> None:
        with pytest.warns(
            QCoDeSDeprecationWarning,
            match="Passing 'instrument' as positional argument",
        ):
            GroupParameter("test", None)

    def test_multiple_positional_args_warn(self) -> None:
        with pytest.warns(
            QCoDeSDeprecationWarning,
            match="'instrument', 'initial_value'",
        ):
            GroupParameter("test", None, None)

    def test_keyword_args_do_not_warn(self) -> None:
        p = GroupParameter("test", instrument=None, initial_value=None)
        assert p.name == "test"

    def test_duplicate_positional_and_keyword_raises(
        self, mock_instrument: _MockInstrument
    ) -> None:
        with pytest.raises(
            TypeError,
            match="got multiple values for argument 'instrument'",
        ):
            GroupParameter("test", None, instrument=mock_instrument)

    def test_too_many_positional_args_raises(self) -> None:
        too_many = (None,) * 5
        with pytest.raises(TypeError, match="takes at most"):
            GroupParameter("test", *too_many)


def _make_delegate_group() -> DelegateGroup:
    """Helper to create a minimal DelegateGroup for testing."""
    source = Parameter("source", set_cmd=None, get_cmd=None)
    dp = DelegateGroupParameter("dp", source=source)
    return DelegateGroup("grp", parameters=[dp])


class TestGroupedParameterPositionalArgs:
    """GroupedParameter should warn when arguments after ``name`` are positional."""

    def test_single_positional_arg_warns(self) -> None:
        grp = _make_delegate_group()
        with pytest.warns(
            QCoDeSDeprecationWarning,
            match="Passing 'group' as positional argument",
        ):
            GroupedParameter("test", grp)

    def test_multiple_positional_args_warn(self) -> None:
        grp = _make_delegate_group()
        with pytest.warns(
            QCoDeSDeprecationWarning,
            match="'group', 'unit'",
        ):
            GroupedParameter("test", grp, "V")

    def test_keyword_args_do_not_warn(self) -> None:
        grp = _make_delegate_group()
        p = GroupedParameter("test", group=grp, unit="V")
        assert p.name == "test"
        assert p.unit == "V"

    def test_duplicate_positional_and_keyword_raises(self) -> None:
        grp = _make_delegate_group()
        with pytest.raises(
            TypeError,
            match="got multiple values for argument 'group'",
        ):
            GroupedParameter("test", grp, group=grp)

    def test_too_many_positional_args_raises(self) -> None:
        too_many = (None,) * 5
        with pytest.raises(TypeError, match="takes at most"):
            GroupedParameter("test", *too_many)

    def test_missing_group_raises(self) -> None:
        with pytest.raises(
            TypeError,
            match="missing required keyword argument: 'group'",
        ):
            GroupedParameter("test")
