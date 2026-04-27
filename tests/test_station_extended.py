"""
Extended tests for Station to improve coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from qcodes.instrument import Instrument, InstrumentModule
from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.parameters import Parameter
from qcodes.station import Station

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(name="station", scope="function")
def _station() -> Iterator[Station]:
    st = Station(default=True)
    try:
        yield st
    finally:
        Station.default = None


@pytest.fixture(name="dummy_instr", scope="function")
def _dummy_instr() -> Iterator[DummyInstrument]:
    inst = DummyInstrument(name="st_dummy", gates=["dac1", "dac2"])
    try:
        yield inst
    finally:
        inst.close()


# ---------------------------------------------------------------------------
# Station.default class attribute
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_station_default_set_on_init() -> None:
    """New Station with default=True should set Station.default."""
    st = Station(default=True)
    assert Station.default is st
    Station.default = None


@pytest.mark.serial
def test_station_default_not_set() -> None:
    """Station with default=False should not overwrite Station.default."""
    Station.default = None
    _ = Station(default=False)
    assert Station.default is None


# ---------------------------------------------------------------------------
# Station.add_component / remove_component
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_add_component(station: Station, dummy_instr: DummyInstrument) -> None:
    name = station.add_component(dummy_instr)
    assert name == "st_dummy"
    assert "st_dummy" in station.components


@pytest.mark.serial
def test_add_component_custom_name(
    station: Station, dummy_instr: DummyInstrument
) -> None:
    name = station.add_component(dummy_instr, name="custom_name")
    assert name == "custom_name"
    assert "custom_name" in station.components


@pytest.mark.serial
def test_add_component_duplicate_raises(
    station: Station, dummy_instr: DummyInstrument
) -> None:
    station.add_component(dummy_instr)
    with pytest.raises(RuntimeError, match="already registered"):
        station.add_component(dummy_instr)


@pytest.mark.serial
def test_remove_component(station: Station, dummy_instr: DummyInstrument) -> None:
    station.add_component(dummy_instr)
    removed = station.remove_component("st_dummy")
    assert removed is dummy_instr
    assert "st_dummy" not in station.components


@pytest.mark.serial
def test_remove_component_not_found(station: Station) -> None:
    with pytest.raises(KeyError, match="is not part of the station"):
        station.remove_component("nonexistent_xyz")


# ---------------------------------------------------------------------------
# Station.__getitem__
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_getitem(station: Station, dummy_instr: DummyInstrument) -> None:
    station.add_component(dummy_instr)
    assert station["st_dummy"] is dummy_instr


@pytest.mark.serial
def test_getitem_missing(station: Station) -> None:
    with pytest.raises(KeyError):
        station["nonexistent"]


# ---------------------------------------------------------------------------
# Station.snapshot_base
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_snapshot_base_empty(station: Station) -> None:
    snap = station.snapshot_base(update=False)
    assert "instruments" in snap
    assert "parameters" in snap
    assert "components" in snap
    assert "config" in snap


@pytest.mark.serial
def test_snapshot_base_with_instrument(
    station: Station, dummy_instr: DummyInstrument
) -> None:
    station.add_component(dummy_instr)
    snap = station.snapshot_base(update=False)
    assert "st_dummy" in snap["instruments"]


@pytest.mark.serial
def test_snapshot_base_with_parameter(station: Station) -> None:
    param = Parameter("standalone_param", set_cmd=None, get_cmd=None, initial_value=5)
    station.add_component(param, name="my_param")
    snap = station.snapshot_base(update=False)
    assert "my_param" in snap["parameters"]


@pytest.mark.serial
def test_snapshot_base_with_other_component(station: Station) -> None:
    """Non-instrument, non-parameter components go into 'components'."""
    # InstrumentModule is Metadatable but not Instrument or Parameter
    instr = DummyInstrument(name="snap_parent_st", gates=["g"])
    try:
        mod = InstrumentModule(instr, "modcomp")
        station.add_component(mod, name="modcomp")
        snap = station.snapshot_base(update=False)
        assert "modcomp" in snap["components"]
    finally:
        instr.close()


@pytest.mark.serial
def test_snapshot_base_removes_closed_instrument(station: Station) -> None:
    """Closed instruments should be removed from station during snapshot."""
    inst = DummyInstrument(name="closing_inst", gates=["g"])
    station.add_component(inst)
    inst.close()
    snap = station.snapshot_base(update=False)
    assert "closing_inst" not in snap["instruments"]
    assert "closing_inst" not in station.components


# ---------------------------------------------------------------------------
# Station.close_all_registered_instruments
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_close_all_registered_instruments(station: Station) -> None:
    d1 = DummyInstrument(name="close_reg_1", gates=["g"])
    d2 = DummyInstrument(name="close_reg_2", gates=["g"])
    station.add_component(d1)
    station.add_component(d2)

    station.close_all_registered_instruments()

    assert "close_reg_1" not in station.components
    assert "close_reg_2" not in station.components
    assert not Instrument.exist("close_reg_1")
    assert not Instrument.exist("close_reg_2")


# ---------------------------------------------------------------------------
# Station.get_component
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_get_component_top_level(
    station: Station, dummy_instr: DummyInstrument
) -> None:
    station.add_component(dummy_instr)
    comp = station.get_component("st_dummy")
    assert comp is dummy_instr


@pytest.mark.serial
def test_get_component_parameter(
    station: Station, dummy_instr: DummyInstrument
) -> None:
    """get_component should resolve sub-components like parameters."""
    station.add_component(dummy_instr)
    comp = station.get_component("st_dummy_dac1")
    assert comp is dummy_instr.parameters["dac1"]


@pytest.mark.serial
def test_get_component_not_found(station: Station) -> None:
    with pytest.raises(KeyError, match="is not part of the station"):
        station.get_component("nonexistent_component")


@pytest.mark.serial
def test_get_component_non_instrumentbase(station: Station) -> None:
    """get_component with remaining parts on a non-InstrumentBase should raise."""
    param = Parameter("toppar", set_cmd=None, get_cmd=None, initial_value=0)
    station.add_component(param, name="toppar")
    with pytest.raises(KeyError, match="no sub-component"):
        station.get_component("toppar_something")


# ---------------------------------------------------------------------------
# Station with components in constructor
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_station_init_with_components() -> None:
    d = DummyInstrument(name="init_comp_st", gates=["g"])
    try:
        st = Station(d)
        assert "init_comp_st" in st.components
    finally:
        d.close()
        Station.default = None


# ---------------------------------------------------------------------------
# Station.delegate_attr_dicts — attribute access to components
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_station_delegate_attr(station: Station, dummy_instr: DummyInstrument) -> None:
    """Station should delegate attribute access to components dict."""
    station.add_component(dummy_instr)
    assert station.st_dummy is dummy_instr  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Station.add_component with snapshot_exclude Parameter
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_add_component_snapshot_exclude_param(station: Station) -> None:
    """snapshot_exclude parameters should not be in snapshot."""
    param = Parameter(
        "hidden_param",
        set_cmd=None,
        get_cmd=None,
        initial_value=42,
        snapshot_exclude=True,
    )
    station.add_component(param, name="hidden_param")
    snap = station.snapshot_base(update=False)
    assert "hidden_param" not in snap["parameters"]
