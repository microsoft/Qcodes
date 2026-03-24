"""Extended tests for qcodes.parameters.grouped_parameter module."""

from __future__ import annotations

import pytest

from qcodes.parameters import Parameter
from qcodes.parameters.grouped_parameter import (
    DelegateGroup,
    DelegateGroupParameter,
    GroupedParameter,
)


def _make_source_and_delegate(
    name: str, initial: float = 0.0
) -> tuple[Parameter, DelegateGroupParameter]:
    """Helper to create a source parameter and a DelegateGroupParameter."""
    source = Parameter(name=f"{name}_source", set_cmd=None, get_cmd=None)
    source.set(initial)
    delegate = DelegateGroupParameter(name=name, source=source)
    return source, delegate


class TestDelegateGroupParameter:
    def test_basic_creation(self) -> None:
        """DelegateGroupParameter wraps a source parameter."""
        source = Parameter(name="src", set_cmd=None, get_cmd=None)
        source.set(3.14)
        dgp = DelegateGroupParameter(name="wrapped", source=source)
        assert dgp.name == "wrapped"
        assert dgp() == 3.14

    def test_set_propagates_to_source(self) -> None:
        """Setting DelegateGroupParameter updates the source."""
        source = Parameter(name="src", set_cmd=None, get_cmd=None)
        dgp = DelegateGroupParameter(name="wrapped", source=source)
        dgp.set(99.0)
        assert source() == 99.0


class TestDelegateGroup:
    def test_get_without_custom_getter(self) -> None:
        """get() returns namedtuple of parameter values by default."""
        _, d1 = _make_source_and_delegate("alpha", 1.0)
        _, d2 = _make_source_and_delegate("beta", 2.0)

        group = DelegateGroup("my_group", parameters=[d1, d2])
        result = group.get()
        assert result.alpha == 1.0
        assert result.beta == 2.0

    def test_get_single_parameter_returns_scalar(self) -> None:
        """get() with single parameter returns scalar, not namedtuple."""
        _, d1 = _make_source_and_delegate("only", 5.0)
        group = DelegateGroup("single_group", parameters=[d1])
        result = group.get()
        assert result == 5.0

    def test_get_with_custom_getter(self) -> None:
        """get() uses custom getter when provided."""
        _, d1 = _make_source_and_delegate("x", 1.0)

        def custom_getter() -> str:
            return "custom_value"

        group = DelegateGroup("cg", parameters=[d1], getter=custom_getter)
        assert group.get() == "custom_value"

    def test_set_with_dict(self) -> None:
        """set() with a dict sets each parameter by name."""
        src_a, d_a = _make_source_and_delegate("a", 0.0)
        src_b, d_b = _make_source_and_delegate("b", 0.0)

        group = DelegateGroup("dict_group", parameters=[d_a, d_b])
        group.set({"a": 10.0, "b": 20.0})
        assert src_a() == 10.0
        assert src_b() == 20.0

    def test_set_single_value_without_setter(self) -> None:
        """set() with a single value broadcasts to all parameters."""
        src_a, d_a = _make_source_and_delegate("a", 0.0)
        src_b, d_b = _make_source_and_delegate("b", 0.0)

        group = DelegateGroup("broadcast_group", parameters=[d_a, d_b])
        group.set(42.0)
        assert src_a() == 42.0
        assert src_b() == 42.0

    def test_set_with_custom_setter(self) -> None:
        """set() uses custom setter when provided."""
        captured: list[object] = []
        _, d1 = _make_source_and_delegate("x", 0.0)

        def custom_setter(value: object) -> None:
            captured.append(value)

        group = DelegateGroup("cs", parameters=[d1], setter=custom_setter)
        group.set("hello")
        assert captured == ["hello"]

    def test_get_parameters(self) -> None:
        """get_parameters() returns formatted result."""
        _, d1 = _make_source_and_delegate("p1", 3.0)
        _, d2 = _make_source_and_delegate("p2", 7.0)

        group = DelegateGroup("gp", parameters=[d1, d2])
        result = group.get_parameters()
        assert result.p1 == 3.0
        assert result.p2 == 7.0

    def test_source_parameters(self) -> None:
        """source_parameters returns tuple of source Parameter objects."""
        src_a, d_a = _make_source_and_delegate("a", 0.0)
        src_b, d_b = _make_source_and_delegate("b", 0.0)

        group = DelegateGroup("sp_group", parameters=[d_a, d_b])
        sources = group.source_parameters
        assert sources == (src_a, src_b)

    def test_custom_formatter(self) -> None:
        """Custom formatter transforms get_parameters output."""
        _, d1 = _make_source_and_delegate("x", 2.0)
        _, d2 = _make_source_and_delegate("y", 3.0)

        def my_fmt(x: float, y: float) -> float:
            return x + y

        group = DelegateGroup("fmt", parameters=[d1, d2], formatter=my_fmt)
        assert group.get() == 5.0

    def test_custom_parameter_names(self) -> None:
        """parameter_names overrides the default names from parameters."""
        _, d1 = _make_source_and_delegate("orig_name", 1.0)

        group = DelegateGroup(
            "named_group", parameters=[d1], parameter_names=["custom_name"]
        )
        assert "custom_name" in group.parameters

    def test_set_from_dict(self) -> None:
        """_set_from_dict sets parameters by name from a dict."""
        src_a, d_a = _make_source_and_delegate("a", 0.0)
        src_b, d_b = _make_source_and_delegate("b", 0.0)

        group = DelegateGroup("sfd", parameters=[d_a, d_b])
        group._set_from_dict({"a": 100.0, "b": 200.0})
        assert src_a() == 100.0
        assert src_b() == 200.0


class TestGroupedParameter:
    def test_basic_creation(self) -> None:
        """GroupedParameter wraps a DelegateGroup."""
        _, d1 = _make_source_and_delegate("ch1", 5.0)
        group = DelegateGroup("g", parameters=[d1])
        gp = GroupedParameter("grouped", group=group)
        assert gp.name == "grouped"
        assert gp.group is group

    def test_repr(self) -> None:
        """__repr__ includes name and source parameters."""
        _src, d1 = _make_source_and_delegate("ch1", 0.0)
        group = DelegateGroup("g", parameters=[d1])
        gp = GroupedParameter("my_grouped", group=group)
        r = repr(gp)
        assert "GroupedParameter" in r
        assert "my_grouped" in r
        assert "source_parameters" in r

    def test_parameters_property(self) -> None:
        """Parameters property returns delegate parameters dict."""
        _, d1 = _make_source_and_delegate("a", 0.0)
        _, d2 = _make_source_and_delegate("b", 0.0)
        group = DelegateGroup("g", parameters=[d1, d2])
        gp = GroupedParameter("gp", group=group)
        assert "a" in gp.parameters
        assert "b" in gp.parameters

    def test_source_parameters_property(self) -> None:
        """source_parameters property delegates to group."""
        src_a, d_a = _make_source_and_delegate("a", 0.0)
        group = DelegateGroup("g", parameters=[d_a])
        gp = GroupedParameter("gp", group=group)
        assert gp.source_parameters == (src_a,)

    def test_get_raw(self) -> None:
        """get_raw returns formatted parameter values."""
        _, d1 = _make_source_and_delegate("v", 42.0)
        group = DelegateGroup("g", parameters=[d1])
        gp = GroupedParameter("gp", group=group)
        assert gp.get_raw() == 42.0

    def test_set_raw(self) -> None:
        """set_raw delegates to group.set."""
        src, d1 = _make_source_and_delegate("v", 0.0)
        group = DelegateGroup("g", parameters=[d1])
        gp = GroupedParameter("gp", group=group)
        gp.set_raw(99.0)
        assert src() == 99.0

    def test_missing_group_raises(self) -> None:
        """GroupedParameter requires group kwarg."""
        with pytest.raises(TypeError, match="missing required keyword argument"):
            GroupedParameter("bad")  # type: ignore[call-arg]

    def test_label_and_unit_defaults(self) -> None:
        """Default label is name, default unit is empty."""
        _, d1 = _make_source_and_delegate("v", 0.0)
        group = DelegateGroup("g", parameters=[d1])
        gp = GroupedParameter("my_param", group=group)
        assert gp.label == "my_param"
        assert gp.unit == ""

    def test_custom_label_and_unit(self) -> None:
        """Custom label and unit are set."""
        _, d1 = _make_source_and_delegate("v", 0.0)
        group = DelegateGroup("g", parameters=[d1])
        gp = GroupedParameter("p", group=group, label="Voltage", unit="V")
        assert gp.label == "Voltage"
        assert gp.unit == "V"
