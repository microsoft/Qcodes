from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.parameters import ManualParameter, ScaledParameter

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(name="instrument")
def _make_instrument() -> "Generator[DummyInstrument, None, None]":
    instrument = DummyInstrument("dummy")
    try:
        target_name = "target_parameter"
        target_label = "Target Parameter"
        target_unit = "V"
        instrument.add_parameter(
            target_name,
            label=target_label,
            unit=target_unit,
            initial_value=1.0,
            get_cmd=None,
            set_cmd=None,
        )
        instrument.scaler = ScaledParameter(  # type: ignore[attr-defined]
            instrument.target_parameter, division=1
        )
        yield instrument
    finally:
        instrument.close()


def test_constructor(instrument: DummyInstrument) -> None:
    # Test the behaviour of the constructor

    # Require a wrapped parameter
    with pytest.raises(TypeError):
        ScaledParameter()  # type: ignore[call-arg]

    # Require a scaling factor
    with pytest.raises(ValueError):
        ScaledParameter(instrument.target_parameter)

    # Require only one scaling factor
    with pytest.raises(ValueError):
        ScaledParameter(instrument.target_parameter, division=1, gain=1)


def test_namelabel(instrument: DummyInstrument) -> None:
    # Test handling of name and label

    # Test correct inheritance
    assert instrument.scaler.name == instrument.target_parameter.name + "_scaled"
    assert instrument.scaler.label == instrument.target_parameter.label + "_scaled"

    # Test correct name/label handling by the constructor
    scaled_name = "scaled"
    scaled_label = "Scaled parameter"
    scaler2 = ScaledParameter(
        instrument.target_parameter, division=1, name=scaled_name, label=scaled_label
    )
    assert scaler2.name == scaled_name
    assert scaler2.label == scaled_label


def test_unit(instrument: DummyInstrument) -> None:
    # Test handling of the units

    # Check if the unit is correctly inherited
    assert instrument.scaler.unit == "V"

    # Check if we can change successfully the unit
    instrument.scaler.unit = "A"
    assert instrument.scaler.unit == "A"

    # Check if unit is correctly set in the constructor
    scaler2 = ScaledParameter(
        instrument.target_parameter, name="scaled_value", division=1, unit="K"
    )
    assert scaler2.unit == "K"


def test_metadata(instrument: DummyInstrument) -> None:
    # Test the metadata

    test_gain = 3
    test_unit = "V"
    instrument.scaler.gain = test_gain
    instrument.scaler.unit = test_unit

    # Check if relevant fields are present in the snapshot
    snap = instrument.scaler.snapshot()
    snap_keys = snap.keys()
    metadata_keys = snap["metadata"].keys()
    assert "division" in snap_keys
    assert "gain" in snap_keys
    assert "role" in snap_keys
    assert "unit" in snap_keys
    assert "variable_multiplier" in metadata_keys
    assert "wrapped_parameter" in metadata_keys
    assert "wrapped_instrument" in metadata_keys

    # Check if the fields are correct
    assert snap["gain"] == test_gain
    assert snap["division"] == 1 / test_gain
    assert snap["role"] == ScaledParameter.Role.GAIN
    assert snap["unit"] == test_unit
    assert snap["metadata"]["variable_multiplier"] is False
    assert snap["metadata"]["wrapped_parameter"] == instrument.target_parameter.name


def test_wrapped_parameter(instrument: DummyInstrument) -> None:
    # Test if the target parameter is correctly inherited
    assert instrument.scaler.wrapped_parameter == instrument.target_parameter


def test_divider(instrument) -> None:
    test_division = 10
    test_value = 5

    instrument.scaler.division = test_division
    instrument.scaler(test_value)
    assert instrument.scaler() == test_value
    assert instrument.target_parameter() == test_division * test_value
    assert instrument.scaler.gain == 1 / test_division
    assert instrument.scaler.role == ScaledParameter.Role.DIVISION


def test_multiplier(instrument: DummyInstrument) -> None:
    test_multiplier = 10
    test_value = 5

    instrument.scaler.gain = test_multiplier
    instrument.scaler(test_value)
    assert instrument.scaler() == test_value
    assert instrument.target_parameter() == test_value / test_multiplier
    assert instrument.scaler.division == 1 / test_multiplier
    assert instrument.scaler.role == ScaledParameter.Role.GAIN


def test_variable_gain(instrument: DummyInstrument) -> None:
    test_value = 5

    initial_gain = 2
    variable_gain_name = "gain"
    gain = ManualParameter(name=variable_gain_name, initial_value=initial_gain)
    instrument.scaler.gain = gain
    instrument.scaler(test_value)

    assert instrument.scaler() == test_value
    assert instrument.target_parameter() == test_value / initial_gain
    assert instrument.scaler.division == 1 / initial_gain

    second_gain = 7
    gain(second_gain)
    # target value must change on scaler value change, not on gain/division
    assert instrument.target_parameter() == test_value / initial_gain
    instrument.scaler(test_value)
    assert instrument.target_parameter() == test_value / second_gain
    assert instrument.scaler.division == 1 / second_gain

    assert instrument.scaler.metadata["variable_multiplier"] == variable_gain_name
