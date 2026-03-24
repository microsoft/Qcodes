"""Extended tests for qcodes.parameters.combined_parameter module."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from qcodes.parameters import Parameter
from qcodes.parameters.combined_parameter import CombinedParameter, combine


@pytest.fixture()
def two_params() -> list[Parameter]:
    return [
        Parameter("x", set_cmd=None, get_cmd=None),
        Parameter("y", set_cmd=None, get_cmd=None),
    ]


class TestCombineFunction:
    def test_combine_creates_combined_parameter(
        self, two_params: list[Parameter]
    ) -> None:
        """combine() convenience function returns CombinedParameter."""
        cp = combine(*two_params, name="xy")
        assert isinstance(cp, CombinedParameter)
        assert cp.dimensionality == 2

    def test_combine_with_label_and_unit(self, two_params: list[Parameter]) -> None:
        """combine() passes label and unit through."""
        cp = combine(*two_params, name="xy", label="X and Y", unit="V")
        # cp.parameter is a parameter like object but these attributes are dynamically added
        assert cp.parameter.label == "X and Y"  # pyright: ignore[reportFunctionMemberAccess]
        assert cp.parameter.unit == "V"  # pyright: ignore[reportFunctionMemberAccess]

    def test_combine_with_aggregator(self, two_params: list[Parameter]) -> None:
        """combine() passes aggregator through."""
        cp = combine(*two_params, name="xy", aggregator=sum)
        assert hasattr(cp, "aggregate")


class TestCombinedParameter:
    def test_set_calls_parameter_sets(self, two_params: list[Parameter]) -> None:
        """set() sets each parameter in order."""
        cp = CombinedParameter(two_params, name="xy")
        swept = cp.sweep(np.array([[1.0, 2.0], [3.0, 4.0]]))
        swept.set(0)
        assert two_params[0]() == 1.0
        assert two_params[1]() == 2.0
        swept.set(1)
        assert two_params[0]() == 3.0
        assert two_params[1]() == 4.0

    def test_aggregate_with_aggregator(self, two_params: list[Parameter]) -> None:
        """_aggregate calls the aggregator function."""

        def my_agg(*vals: int) -> int:
            return sum(vals)

        cp = CombinedParameter(two_params, name="xy", aggregator=my_agg)
        result = cp._aggregate(1, 2, 3)
        assert result == 6

    def test_aggregate_without_aggregator(self, two_params: list[Parameter]) -> None:
        """Without aggregator, _aggregate is not set as 'aggregate' attr."""
        cp = CombinedParameter(two_params, name="xy")
        assert not hasattr(cp, "aggregate")

    def test_iter(self, two_params: list[Parameter]) -> None:
        """__iter__ iterates over setpoint indices."""
        cp = CombinedParameter(two_params, name="xy")
        swept = cp.sweep(np.array([[1, 2], [3, 4], [5, 6]]))
        indices = list(swept)
        assert indices == [0, 1, 2]

    def test_len(self, two_params: list[Parameter]) -> None:
        """__len__ returns number of setpoints."""
        cp = CombinedParameter(two_params, name="xy")
        swept = cp.sweep(np.array([[1, 2], [3, 4]]))
        assert len(swept) == 2

    def test_len_no_setpoints(self, two_params: list[Parameter]) -> None:
        """__len__ returns 0 when no setpoints."""
        cp = CombinedParameter(two_params, name="xy")
        assert len(cp) == 0

    def test_snapshot_base(self, two_params: list[Parameter]) -> None:
        """snapshot_base returns dict with expected keys."""
        cp = CombinedParameter(two_params, name="xy", label="combined", unit="mV")
        snap = cp.snapshot_base()
        assert snap["label"] == "combined"
        assert snap["unit"] == "mV"
        assert snap["full_name"] == "xy"
        assert "__class__" in snap
        assert "aggregator" in snap

    def test_snapshot_base_with_aggregator(self, two_params: list[Parameter]) -> None:
        """snapshot_base includes aggregator repr."""
        cp = CombinedParameter(two_params, name="xy", aggregator=sum)
        snap = cp.snapshot_base()
        assert "sum" in snap["aggregator"]

    def test_units_deprecated(
        self, two_params: list[Parameter], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Passing units= triggers a deprecation warning log."""
        with caplog.at_level(logging.WARNING):
            cp = CombinedParameter(two_params, name="xy", units="mV")
        assert any("`units` is deprecated" in msg for msg in caplog.messages)
        # cp.parameter is a parameter like object but these attributes are dynamically added
        assert cp.parameter.unit == "mV"  # pyright: ignore[reportFunctionMemberAccess]

    def test_units_deprecated_unit_takes_precedence(
        self, two_params: list[Parameter], caplog: pytest.LogCaptureFixture
    ) -> None:
        """When both unit and units are given, unit takes precedence."""
        with caplog.at_level(logging.WARNING):
            cp = CombinedParameter(two_params, name="xy", unit="V", units="mV")
        # cp.parameter is a parameter like object but these attributes are dynamically added
        assert cp.parameter.unit == "V"  # pyright: ignore[reportFunctionMemberAccess]

    def test_invalid_name_raises(self, two_params: list[Parameter]) -> None:
        """Invalid parameter name raises ValueError."""
        with pytest.raises(ValueError, match="valid identifier"):
            CombinedParameter(two_params, name="invalid name")

    def test_sweep_multiple_arrays(self, two_params: list[Parameter]) -> None:
        """sweep() with multiple 1D arrays."""
        cp = CombinedParameter(two_params, name="xy")
        swept = cp.sweep(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert len(swept) == 3
        swept.set(0)
        assert two_params[0]() == 1
        assert two_params[1]() == 4
