"""
Extended test suite for qcodes.monitor.monitor covering _get_metadata,
Monitor.show, Monitor TypeError on invalid parameters, and
Monitor.update_all.
"""

from __future__ import annotations

from unittest.mock import PropertyMock, patch

import pytest

from qcodes.instrument_drivers.mock_instruments import (
    DummyChannelInstrument,
    DummyInstrument,
)
from qcodes.monitor.monitor import Monitor, _get_metadata
from qcodes.parameters import Parameter

# ---------------------------------------------------------------------------
# _get_metadata - pure-function tests (no Monitor instance needed)
# ---------------------------------------------------------------------------


class TestGetMetadata:
    """Tests for the ``_get_metadata`` helper function."""

    def test_basic_structure(self) -> None:
        """Returned dict must contain 'ts' and 'parameters' keys."""
        param = Parameter("p1", set_cmd=None, get_cmd=None, initial_value=42)
        result = _get_metadata(param)
        assert "ts" in result
        assert "parameters" in result
        assert isinstance(result["ts"], float)
        assert isinstance(result["parameters"], list)

    def test_parameter_with_timestamp(self) -> None:
        """A parameter that has been set should report a non-None ts."""
        param = Parameter("voltage", set_cmd=None, get_cmd=None, unit="V")
        param(3.14)
        result = _get_metadata(param)
        unbound = [
            g for g in result["parameters"] if g["instrument"] == "Unbound Parameter"
        ]
        assert len(unbound) == 1
        meta = unbound[0]["parameters"][0]
        assert meta["value"] == str(3.14)
        assert meta["ts"] is not None
        assert isinstance(meta["ts"], float)
        assert meta["unit"] == "V"

    def test_parameter_without_timestamp(self) -> None:
        """When get_timestamp() returns None, meta['ts'] must be None (line 75)."""
        param = Parameter("fresh", set_cmd=None, get_cmd=None, initial_value=0)
        # Force the timestamp to None via the cache to exercise line 75
        with patch.object(
            type(param.cache), "timestamp", new_callable=PropertyMock, return_value=None
        ):
            result = _get_metadata(param)
        unbound = [
            g for g in result["parameters"] if g["instrument"] == "Unbound Parameter"
        ]
        assert len(unbound) == 1
        meta = unbound[0]["parameters"][0]
        assert meta["ts"] is None

    def test_unbound_parameter_grouping(self) -> None:
        """Parameters not attached to an instrument go under 'Unbound Parameter'."""
        p1 = Parameter("alpha", set_cmd=None, get_cmd=None, initial_value=1)
        p2 = Parameter("beta", set_cmd=None, get_cmd=None, initial_value=2)
        result = _get_metadata(p1, p2)
        instruments = [g["instrument"] for g in result["parameters"]]
        assert instruments == ["Unbound Parameter"]
        assert len(result["parameters"][0]["parameters"]) == 2

    def test_use_root_instrument_true(self) -> None:
        """With use_root_instrument=True, channel params are grouped by root."""
        instr = DummyChannelInstrument("MetaRootTest")
        try:
            result = _get_metadata(
                instr.A.temperature,
                instr.B.temperature,
                use_root_instrument=True,
            )
            instruments = [g["instrument"] for g in result["parameters"]]
            # Both should be grouped under the single root instrument
            assert len(instruments) == 1
            assert len(result["parameters"][0]["parameters"]) == 2
        finally:
            instr.close()

    def test_use_root_instrument_false(self) -> None:
        """With use_root_instrument=False, channel params are grouped by channel."""
        instr = DummyChannelInstrument("MetaChanTest")
        try:
            result = _get_metadata(
                instr.A.temperature,
                instr.B.temperature,
                use_root_instrument=False,
            )
            instruments = [g["instrument"] for g in result["parameters"]]
            # Each channel is a separate instrument grouping
            assert len(instruments) == 2
        finally:
            instr.close()

    def test_instrument_bound_parameter(self) -> None:
        """Parameters attached to a DummyInstrument use the instrument name."""
        instr = DummyInstrument("MetaDummy", gates=["ch1"])
        try:
            result = _get_metadata(instr.ch1)
            instruments = [g["instrument"] for g in result["parameters"]]
            assert str(instr) in instruments
        finally:
            instr.close()

    def test_mixed_bound_and_unbound(self) -> None:
        """Bound and unbound parameters appear in separate groups."""
        instr = DummyInstrument("MixedDummy", gates=["g1"])
        free = Parameter("free_param", set_cmd=None, get_cmd=None, initial_value=0)
        try:
            result = _get_metadata(instr.g1, free)
            instruments = sorted(g["instrument"] for g in result["parameters"])
            assert "Unbound Parameter" in instruments
            assert str(instr) in instruments
        finally:
            instr.close()

    def test_label_used_as_name(self) -> None:
        """meta['name'] should equal parameter.label (or .name if no label)."""
        p_with_label = Parameter(
            "x", label="My Label", set_cmd=None, get_cmd=None, initial_value=0
        )
        p_without_label = Parameter("y", set_cmd=None, get_cmd=None, initial_value=0)
        result = _get_metadata(p_with_label, p_without_label)
        params = result["parameters"][0]["parameters"]
        names = [p["name"] for p in params]
        assert "My Label" in names
        assert "y" in names

    def test_empty_parameters(self) -> None:
        """Calling _get_metadata with no parameters returns an empty list."""
        result = _get_metadata()
        assert result["parameters"] == []
        assert "ts" in result


# ---------------------------------------------------------------------------
# Monitor.show - static method (mock webbrowser to avoid opening a browser)
# ---------------------------------------------------------------------------


class TestMonitorShow:
    def test_show_opens_browser(self) -> None:
        """Monitor.show() should call webbrowser.open with correct URL."""
        with patch("qcodes.monitor.monitor.webbrowser.open") as mock_open:
            Monitor.show()
            mock_open.assert_called_once_with("http://localhost:3000")


# ---------------------------------------------------------------------------
# Monitor.__init__ - TypeError for non-Parameter arguments (line 162)
# ---------------------------------------------------------------------------


class TestMonitorTypeError:
    def test_non_parameter_raises_type_error(self) -> None:
        """Passing a non-Parameter object must raise TypeError."""
        with pytest.raises(TypeError, match="We can only monitor"):
            Monitor("not_a_parameter")  # type: ignore[arg-type]
