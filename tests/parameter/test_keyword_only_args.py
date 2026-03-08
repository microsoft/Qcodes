"""Tests that passing positional arguments (beyond ``name``) to
parameter classes is rejected now that *args support has been removed.
"""

from __future__ import annotations

from typing import Any

import pytest

from qcodes.parameters import (
    ArrayParameter,
    ElapsedTimeParameter,
    GroupParameter,
    InstrumentRefParameter,
    MultiParameter,
    Parameter,
    ParameterBase,
)
from qcodes.parameters.grouped_parameter import (
    DelegateGroup,
    DelegateGroupParameter,
    GroupedParameter,
)
from qcodes.parameters.multi_channel_instrument_parameter import (
    MultiChannelInstrumentParameter,
)


# Minimal concrete subclass of ParameterBase for testing
class _ConcreteParameterBase(ParameterBase):
    def get_raw(self) -> Any:
        return 0


class TestParameterBaseKeywordOnly:
    """ParameterBase requires keyword-only arguments after ``name``."""

    def test_keyword_args_work(self) -> None:
        p = _ConcreteParameterBase("test", instrument=None, snapshot_get=True)
        assert p.name == "test"

    def test_positional_args_rejected(self) -> None:
        with pytest.raises(TypeError):
            _ConcreteParameterBase("test", None)  # type: ignore[misc]


class TestParameterKeywordOnly:
    """Parameter requires keyword-only arguments after ``name``."""

    def test_keyword_args_work(self) -> None:
        p = Parameter("test", instrument=None, label="my label", set_cmd=None)
        assert p.name == "test"
        assert p.label == "my label"

    def test_positional_args_rejected(self) -> None:
        with pytest.raises(TypeError):
            Parameter("test", None, set_cmd=None)  # type: ignore[misc]


# Minimal concrete subclass of ArrayParameter for testing
class _ConcreteArrayParameter(ArrayParameter):
    def get_raw(self) -> Any:
        return [0]


class TestArrayParameterKeywordOnly:
    """ArrayParameter requires keyword-only arguments after ``name``."""

    def test_keyword_args_work(self) -> None:
        p = _ConcreteArrayParameter("test", shape=(3,), instrument=None)
        assert p.name == "test"
        assert p.shape == (3,)

    def test_positional_args_rejected(self) -> None:
        with pytest.raises(TypeError):
            _ConcreteArrayParameter("test", (3,))  # type: ignore[misc]

    def test_missing_shape_raises(self) -> None:
        with pytest.raises(TypeError):
            _ConcreteArrayParameter("test")  # type: ignore[misc]


class TestGroupParameterKeywordOnly:
    """GroupParameter requires keyword-only arguments after ``name``."""

    def test_keyword_args_work(self) -> None:
        p = GroupParameter("test", instrument=None, initial_value=None)
        assert p.name == "test"

    def test_positional_args_rejected(self) -> None:
        with pytest.raises(TypeError):
            GroupParameter("test", None)  # type: ignore[misc]


def _make_delegate_group() -> DelegateGroup:
    """Helper to create a minimal DelegateGroup for testing."""
    source = Parameter("source", set_cmd=None, get_cmd=None)
    dp = DelegateGroupParameter("dp", source)
    return DelegateGroup("grp", parameters=[dp])


class TestGroupedParameterKeywordOnly:
    """GroupedParameter requires keyword-only arguments after ``name``."""

    def test_keyword_args_work(self) -> None:
        grp = _make_delegate_group()
        p = GroupedParameter("test", group=grp, unit="V")
        assert p.name == "test"
        assert p.unit == "V"

    def test_positional_args_rejected(self) -> None:
        grp = _make_delegate_group()
        with pytest.raises(TypeError):
            GroupedParameter("test", grp)  # type: ignore[misc]

    def test_missing_group_raises(self) -> None:
        with pytest.raises(TypeError):
            GroupedParameter("test")  # type: ignore[misc]


# Minimal concrete subclass of MultiParameter for testing
class _ConcreteMultiParameter(MultiParameter):
    def get_raw(self) -> Any:
        return (0,)


class TestMultiParameterKeywordOnly:
    """MultiParameter requires keyword-only arguments after ``name``."""

    def test_keyword_args_work(self) -> None:
        p = _ConcreteMultiParameter("test", names=("a",), shapes=((),))
        assert p.name == "test"
        assert p.names == ("a",)

    def test_positional_args_rejected(self) -> None:
        with pytest.raises(TypeError):
            _ConcreteMultiParameter("test", ("a",), ((),))  # type: ignore[misc]

    def test_missing_names_raises(self) -> None:
        with pytest.raises(TypeError):
            _ConcreteMultiParameter("test", shapes=((),))  # type: ignore[misc]

    def test_missing_shapes_raises(self) -> None:
        with pytest.raises(TypeError):
            _ConcreteMultiParameter("test", names=("a",))  # type: ignore[misc]


class TestMultiChannelInstrumentParameterKeywordOnly:
    """MultiChannelInstrumentParameter requires keyword-only arguments."""

    def test_keyword_args_work(self) -> None:
        p = MultiChannelInstrumentParameter(
            channels=[], param_name="x", name="test", names=("a",), shapes=((),)
        )
        assert p.name == "test"

    def test_positional_args_rejected(self) -> None:
        with pytest.raises(TypeError):
            MultiChannelInstrumentParameter(
                [],  # pyright: ignore[reportCallIssue]
                "x",
                name="test",
                names=("a",),
                shapes=((),),
            )

    def test_missing_channels_raises(self) -> None:
        with pytest.raises(TypeError):
            MultiChannelInstrumentParameter(  # type: ignore[call-arg]
                param_name="x", name="test", names=("a",), shapes=((),)
            )

    def test_missing_param_name_raises(self) -> None:
        with pytest.raises(TypeError):
            MultiChannelInstrumentParameter(  # type: ignore[call-arg]
                channels=[], name="test", names=("a",), shapes=((),)
            )


class TestElapsedTimeParameterKeywordOnly:
    """ElapsedTimeParameter requires keyword-only arguments after ``name``."""

    def test_keyword_args_work(self) -> None:
        p = ElapsedTimeParameter("test", label="My label")
        assert p.name == "test"
        assert p.label == "My label"

    def test_positional_args_rejected(self) -> None:
        with pytest.raises(TypeError):
            ElapsedTimeParameter("test", "My label")  # type: ignore[misc]


class TestInstrumentRefParameterKeywordOnly:
    """InstrumentRefParameter requires keyword-only arguments after ``name``."""

    def test_keyword_args_work(self) -> None:
        p = InstrumentRefParameter("test", instrument=None, label="my label")
        assert p.name == "test"
        assert p.label == "my label"

    def test_positional_args_rejected(self) -> None:
        with pytest.raises(TypeError):
            InstrumentRefParameter("test", None)  # type: ignore[misc]
