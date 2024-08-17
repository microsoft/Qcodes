import math
from unittest.mock import MagicMock

from qcodes.instrument_drivers.Keysight.keysightb1500.constants import (
    DCORR,
    IMP,
    SlotNr,
)
from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500_module import (
    _DCORRResponse,
    _FMTResponse,
    convert_dummy_val_to_nan,
    fixed_negative_float,
    format_dcorr_response,
    get_name_label_unit_of_impedance_model,
    parse_module_query_response,
)
from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1517A import (
    KeysightB1517A,
)


def test_is_enabled() -> None:
    mainframe = MagicMock()

    # Use concrete subclass because B1500Module does not assign channels
    smu = KeysightB1517A(parent=mainframe, name="B1517A", slot_nr=1)

    mainframe.ask.return_value = "CN 1,2,4,8"
    assert smu.is_enabled()
    mainframe.ask.assert_called_once_with("*LRN? 0")

    mainframe.reset_mock(return_value=True)
    mainframe.ask.return_value = "CN 2,4,8"
    assert not smu.is_enabled()
    mainframe.ask.assert_called_once_with("*LRN? 0")

    mainframe.reset_mock(return_value=True)
    mainframe.ask.return_value = "CN"
    assert not smu.is_enabled()
    mainframe.ask.assert_called_once_with("*LRN? 0")


def test_enable_outputs() -> None:
    mainframe = MagicMock()

    slot_nr = 1
    # Use concrete subclass because B1500Module does not assign channels
    smu = KeysightB1517A(parent=mainframe, name="B1517A", slot_nr=slot_nr)

    smu.enable_outputs()
    mainframe.write.assert_called_once_with(f"CN {slot_nr}")


def test_disable_outputs() -> None:
    mainframe = MagicMock()

    slot_nr = 1
    # Use concrete subclass because B1500Module does not assign channels
    smu = KeysightB1517A(parent=mainframe, name="B1517A", slot_nr=slot_nr)

    smu.disable_outputs()
    mainframe.write.assert_called_once_with(f"CL {slot_nr}")


def test_parse_module_query_response() -> None:
    response = "B1517A,0;B1517A,0;B1520A,0;0,0;0,0;0,0;0,0;0,0;0,0;0,0"
    expected = {
        SlotNr.SLOT01: "B1517A",
        SlotNr.SLOT02: "B1517A",
        SlotNr.SLOT03: "B1520A",
    }

    actual = parse_module_query_response(response)

    assert actual == expected


def test_format_dcorr_response() -> None:
    resp_str1 = format_dcorr_response(
        _DCORRResponse(mode=DCORR.Mode.Cp_G, primary=0.001, secondary=0.34)
    )
    assert resp_str1 == "Mode: Cp_G, Primary Cp: 0.001 F, Secondary G: 0.34 S"

    resp_str2 = format_dcorr_response(
        _DCORRResponse(mode=DCORR.Mode.Ls_Rs, primary=0.2, secondary=3.0)
    )
    assert resp_str2 == "Mode: Ls_Rs, Primary Ls: 0.2 H, Secondary Rs: 3.0 Ω"


def test_fixed_negative_float() -> None:
    assert fixed_negative_float("-0.-1") == -0.1
    assert fixed_negative_float("-1.-1") == -1.1
    assert fixed_negative_float("0.1") == 0.1
    assert fixed_negative_float("1.0") == 1.0
    assert fixed_negative_float("1") == 1.0
    assert fixed_negative_float("-1") == -1.0


def test_get_name_label_unit_of_impedance_model() -> None:
    model = IMP.MeasurementMode.Cp_D
    name, label, unit = get_name_label_unit_of_impedance_model(model)
    assert name == ("parallel_capacitance", "dissipation_factor")
    assert label == ("Parallel Capacitance", "Dissipation Factor")
    assert unit == ("F", "")

    model = IMP.MeasurementMode.Y_THETA_DEG
    name, label, unit = get_name_label_unit_of_impedance_model(model)
    assert name == ("admittance", "phase")
    assert label == ("Admittance", "Phase")
    assert unit == ("S", "degree")


def test_convert_dummy_val_to_nan() -> None:
    status = ["C", "V", "N", "V", "N", "N"]
    value = [0, 199.999e99, 1, 199.999e99, 2, 3]
    channel = [1, 1, 1, 1, 1, 1]
    param_type = ["V", "V", "V", "V", "V", "V"]
    param = _FMTResponse(value, status, channel, param_type)
    convert_dummy_val_to_nan(param)
    assert math.isnan(param.value[1])
    assert math.isnan(param.value[3])
