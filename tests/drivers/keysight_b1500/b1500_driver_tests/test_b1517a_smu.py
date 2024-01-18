import math
import re
from unittest.mock import MagicMock, call

import pytest

from qcodes.instrument_drivers.Keysight.keysightb1500 import constants
from qcodes.instrument_drivers.Keysight.keysightb1500.constants import (
    MM,
    CompliancePolarityMode,
    IMeasRange,
    IOutputRange,
    VMeasRange,
    VOutputRange,
)
from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1517A import (
    KeysightB1517A,
)


@pytest.fixture(name="smu")
def _make_smu(mainframe):
    slot_nr = 1
    smu = KeysightB1517A(parent=mainframe, name="B1517A", slot_nr=slot_nr)
    yield smu


def test_snapshot() -> None:
    from qcodes.instrument.base import InstrumentBase

    # We need to use `InstrumentBase` (not a bare mock) in order for
    # `snapshot` methods call resolution to work out
    mainframe = InstrumentBase(name='mainframe')
    mainframe.write = MagicMock()  # type: ignore[attr-defined]
    slot_nr = 1
    smu = KeysightB1517A(
        parent=mainframe, name="B1517A", slot_nr=slot_nr  # type: ignore[arg-type]
    )

    smu.use_high_speed_adc()
    smu.source_config(output_range=VOutputRange.AUTO)
    smu.i_measure_range_config(i_measure_range=IMeasRange.AUTO)
    smu.v_measure_range_config(v_measure_range=VMeasRange.AUTO)
    smu.timing_parameters(0.0, 0.123, 321)

    s = smu.snapshot()

    assert '_source_config' in s
    assert 'output_range' in s['_source_config']
    assert isinstance(s['_source_config']['output_range'], VOutputRange)
    assert '_measure_config' in s
    assert 'v_measure_range' in s['_measure_config']
    assert 'i_measure_range' in s['_measure_config']
    assert isinstance(s['_measure_config']['v_measure_range'], VMeasRange)
    assert isinstance(s['_measure_config']['i_measure_range'], IMeasRange)
    assert '_timing_parameters' in s
    assert 'number' in s['_timing_parameters']
    assert isinstance(s['_timing_parameters']['number'], int)


def test_v_measure_range_config_raises_type_error(smu) -> None:
    msg = re.escape("Expected valid voltage measurement range, got 42.")

    with pytest.raises(TypeError, match=msg):
        smu.v_measure_range_config(v_measure_range=42)


def test_v_measure_range_config_raises_invalid_range_error(smu) -> None:
    msg = re.escape("15000 voltage measurement range")
    with pytest.raises(RuntimeError, match=msg):
        smu.v_measure_range_config(VMeasRange.MIN_1500V)


def test_v_measure_range_config_sets_range_correctly(smu) -> None:
    smu.v_measure_range_config(v_measure_range=VMeasRange.MIN_0V5)
    s = smu.snapshot()

    assert isinstance(s['_measure_config']['v_measure_range'], VMeasRange)
    assert s['_measure_config']['v_measure_range'] == 5


def test_getting_voltage_after_calling_v_measure_range_config(smu) -> None:
    mainframe = smu.parent
    mainframe.ask.return_value = "NAV-000.002E-01\r"

    smu.v_measure_range_config(VMeasRange.FIX_2V)

    assert smu.voltage.measurement_status is None
    assert pytest.approx(-0.2e-3) == smu.voltage()
    assert smu.voltage.measurement_status == constants.MeasurementStatus.N

    s = smu.voltage.snapshot()
    assert s


def test_i_measure_range_config_raises_type_error(smu) -> None:
    msg = re.escape("Expected valid current measurement range, got 99.")

    with pytest.raises(TypeError, match=msg):
        smu.i_measure_range_config(i_measure_range=99)


def test_i_measure_range_config_raises_invalid_range_error(smu) -> None:
    msg = re.escape("-23 current measurement range")
    with pytest.raises(RuntimeError, match=msg):
        smu.i_measure_range_config(IMeasRange.FIX_40A)


def test_i_measure_range_config_sets_range_correctly(smu) -> None:
    smu.i_measure_range_config(i_measure_range=IMeasRange.MIN_1nA)
    s = smu.snapshot()

    assert isinstance(s['_measure_config']['i_measure_range'], IMeasRange)
    assert s['_measure_config']['i_measure_range'] == 11


def test_getting_current_after_calling_i_measure_range_config(smu) -> None:
    mainframe = smu.parent
    mainframe.ask.return_value = "NAI+000.005E-06\r"

    smu.i_measure_range_config(IMeasRange.MIN_100mA)

    assert smu.current.measurement_status is None
    assert pytest.approx(0.005e-6) == smu.current()
    assert smu.current.measurement_status == constants.MeasurementStatus.N

    s = smu.current.snapshot()
    assert s


def test_force_invalid_voltage_output_range(smu) -> None:
    msg = re.escape("Invalid Source Voltage Output Range")
    with pytest.raises(RuntimeError, match=msg):
        smu.source_config(VOutputRange.MIN_1500V)


def test_force_invalid_current_output_range(smu) -> None:
    msg = re.escape("Invalid Source Current Output Range")
    with pytest.raises(RuntimeError, match=msg):
        smu.source_config(IOutputRange.MIN_20A)


def test_force_voltage_with_autorange(smu) -> None:
    mainframe = smu.parent

    smu.source_config(output_range=VOutputRange.AUTO)
    smu.voltage(10)

    mainframe.write.assert_called_once_with('DV 1,0,10')


def test_force_voltage_autorange_and_compliance(smu) -> None:
    mainframe = smu.parent

    smu.source_config(output_range=VOutputRange.AUTO,
                      compliance=1e-6,
                      compl_polarity=CompliancePolarityMode.AUTO,
                      min_compliance_range=IOutputRange.MIN_10uA)
    smu.voltage(20)

    mainframe.write.assert_called_once_with('DV 1,0,20,1e-06,0,15')


def test_new_source_config_should_invalidate_old_source_config(smu) -> None:
    mainframe = smu.parent

    smu.source_config(output_range=VOutputRange.AUTO,
                      compliance=1e-6,
                      compl_polarity=CompliancePolarityMode.AUTO,
                      min_compliance_range=IOutputRange.MIN_10uA)

    smu.source_config(output_range=VOutputRange.AUTO)
    smu.voltage(20)

    mainframe.write.assert_called_once_with('DV 1,0,20')


def test_unconfigured_source_defaults_to_autorange_v(smu) -> None:
    mainframe = smu.parent

    smu.voltage(10)

    mainframe.write.assert_called_once_with('DV 1,0,10')


def test_unconfigured_source_defaults_to_autorange_i(smu) -> None:
    mainframe = smu.parent

    smu.current(0.2)

    mainframe.write.assert_called_once_with('DI 1,0,0.2')


def test_force_current_with_autorange(smu) -> None:
    mainframe = smu.parent

    smu.source_config(output_range=IOutputRange.AUTO)
    smu.current(0.1)

    mainframe.write.assert_called_once_with('DI 1,0,0.1')


def test_raise_warning_output_range_mismatches_output_command(smu) -> None:
    smu.source_config(output_range=VOutputRange.AUTO)
    msg = re.escape("Asking to force current, but source_config contains a "
                    "voltage output range")
    with pytest.raises(TypeError, match=msg):
        smu.current(0.1)

    smu.source_config(output_range=IOutputRange.AUTO)
    msg = re.escape("Asking to force voltage, but source_config contains a "
                    "current output range")
    with pytest.raises(TypeError, match=msg):
        smu.voltage(0.1)


def test_measure_current(smu) -> None:
    mainframe = smu.parent
    mainframe.ask.return_value = "NAI+000.005E-06\r"

    assert smu.current.measurement_status is None

    assert pytest.approx(0.005e-6) == smu.current()
    assert smu.current.measurement_status == constants.MeasurementStatus.N


def test_measure_voltage(smu) -> None:
    mainframe = smu.parent
    mainframe.ask.return_value = "NAV+000.123E-06\r"

    assert smu.voltage.measurement_status is None

    assert pytest.approx(0.123e-6) == smu.voltage()
    assert smu.voltage.measurement_status == constants.MeasurementStatus.N

    s = smu.voltage.snapshot()
    assert s


def test_measure_current_shows_compliance_hit(smu) -> None:
    mainframe = smu.parent
    mainframe.ask.return_value = "CAI+000.123E-06\r"

    assert smu.current.measurement_status is None

    assert pytest.approx(0.123e-6) == smu.current()
    assert smu.current.measurement_status == constants.MeasurementStatus.C


def test_measured_voltage_with_V_status_returns_nan(smu) -> None:
    mainframe = smu.parent
    mainframe.ask.return_value = "VAV+199.999E+99\r"

    assert smu.voltage.measurement_status is None

    assert math.isnan(smu.voltage())
    assert smu.voltage.measurement_status == constants.MeasurementStatus.V


def test_some_voltage_sourcing_and_current_measurement(smu) -> None:
    mainframe = smu.parent

    smu.source_config(output_range=VOutputRange.MIN_0V5, compliance=1e-9)
    smu.i_measure_range_config(IMeasRange.FIX_100nA)

    mainframe.ask.return_value = "NAI+000.005E-09\r"

    smu.voltage(6)

    mainframe.write.assert_called_once_with('DV 1,5,6,1e-09')

    assert pytest.approx(0.005e-9) == smu.current()

    assert smu.voltage.measurement_status is None
    assert smu.current.measurement_status == constants.MeasurementStatus.N


def test_use_high_resolution_adc(smu) -> None:
    mainframe = smu.parent

    smu.use_high_resolution_adc()

    mainframe.write.assert_called_once_with('AAD 1,1')


def test_use_high_speed_adc(smu) -> None:
    mainframe = smu.parent

    smu.use_high_speed_adc()

    mainframe.write.assert_called_once_with('AAD 1,0')


def test_measurement_mode_at_init(smu) -> None:
    mode_at_init = smu.measurement_mode()
    assert mode_at_init == MM.Mode.SPOT


def test_measurement_mode_to_enum_value(smu) -> None:
    mainframe = smu.parent

    smu.measurement_mode(MM.Mode.SAMPLING)
    mainframe.write.assert_called_once_with('MM 10,1')

    new_mode = smu.measurement_mode()
    assert new_mode == MM.Mode.SAMPLING


def test_measurement_mode_to_int_value(smu) -> None:
    mainframe = smu.parent

    smu.measurement_mode(10)
    mainframe.write.assert_called_once_with('MM 10,1')

    new_mode = smu.measurement_mode()
    assert new_mode == MM.Mode.SAMPLING


def test_setting_timing_parameters(smu) -> None:
    mainframe = smu.parent

    smu.timing_parameters(0.0, 0.42, 32)
    mainframe.write.assert_called_once_with('MT 0.0,0.42,32')

    mainframe.reset_mock()

    smu.timing_parameters(0.0, 0.42, 32, 0.02)
    mainframe.write.assert_called_once_with('MT 0.0,0.42,32,0.02')


def test_set_average_samples_for_high_speed_adc(smu) -> None:
    mainframe = smu.parent

    smu.set_average_samples_for_high_speed_adc(131, 2)
    mainframe.write.assert_called_once_with('AV 131,2')

    mainframe.reset_mock()

    smu.set_average_samples_for_high_speed_adc(132)
    mainframe.write.assert_called_once_with('AV 132,0')



def test_measurement_operation_mode(smu) -> None:
    mainframe = smu.parent

    smu.measurement_operation_mode(constants.CMM.Mode.COMPLIANCE_SIDE)
    mainframe.write.assert_called_once_with('CMM 1,0')

    mainframe.reset_mock()

    mainframe.ask.return_value = 'CMM 1,0'
    cmm_mode = smu.measurement_operation_mode()
    assert cmm_mode == [(constants.ChNr.SLOT_01_CH1,
                         constants.CMM.Mode.COMPLIANCE_SIDE)]


def test_current_measurement_range(smu) -> None:
    mainframe = smu.parent

    smu.current_measurement_range(constants.IMeasRange.FIX_1A)
    mainframe.write.assert_called_once_with('RI 1,-20')

    mainframe.reset_mock()

    mainframe.ask.return_value = 'RI 1,-20'
    cmm_mode = smu.current_measurement_range()
    assert cmm_mode == [(constants.ChNr.SLOT_01_CH1,
                         constants.IMeasRange.FIX_1A)]


def test_get_sweep_mode_range_start_end_steps(smu) -> None:
    mainframe = smu.parent
    mainframe.ask.return_value = 'WV1,1,50,+3.0E+00,-3.0E+00,201'

    sweep_mode = smu.iv_sweep.sweep_mode()
    assert constants.SweepMode(1) == sweep_mode

    mainframe.reset_mock()

    sweep_range = smu.iv_sweep.sweep_range()
    assert constants.VOutputRange(50) == sweep_range

    sweep_start = smu.iv_sweep.sweep_start()
    assert 3.0 == sweep_start

    sweep_start = smu.iv_sweep.sweep_end()
    assert -3.0 == sweep_start

    sweep_start = smu.iv_sweep.sweep_steps()
    assert 201 == sweep_start

    current_compliance = smu.iv_sweep.current_compliance()
    assert current_compliance is None


def test_iv_sweep_delay(smu) -> None:
    mainframe = smu.root_instrument

    smu.iv_sweep.hold_time(43.12)
    smu.iv_sweep.delay(34.01)
    smu.iv_sweep.step_delay(0.01)
    smu.iv_sweep.trigger_delay(0.1)
    smu.iv_sweep.measure_delay(15.4)

    mainframe.write.assert_has_calls([call("WT 43.12,0.0,0.0,0.0,0.0"),
                                      call("WT 43.12,34.01,0.0,0.0,0.0"),
                                      call("WT 43.12,34.01,0.01,0.0,0.0"),
                                      call("WT 43.12,34.01,0.01,0.1,0.0"),
                                      call("WT 43.12,34.01,0.01,0.1,15.4")])


def test_iv_sweep_mode_start_end_steps_compliance(smu) -> None:
    mainframe = smu.parent

    smu.iv_sweep.sweep_mode(constants.SweepMode.LINEAR_TWO_WAY)
    smu.iv_sweep.sweep_range(constants.VOutputRange.MIN_2V)
    smu.iv_sweep.sweep_start(0.2)
    smu.iv_sweep.sweep_end(12.3)
    smu.iv_sweep.sweep_steps(13)
    smu.iv_sweep.current_compliance(45e-3)
    smu.iv_sweep.power_compliance(0.2)

    mainframe.write.assert_has_calls([call('WV 1,3,0,0.0,0.0,1'),
                                      call('WV 1,3,20,0.0,0.0,1'),
                                      call('WV 1,3,20,0.2,0.0,1'),
                                      call('WV 1,3,20,0.2,12.3,1'),
                                      call('WV 1,3,20,0.2,12.3,13'),
                                      call('WV 1,3,20,0.2,12.3,13,0.045'),
                                      call('WV 1,3,20,0.2,12.3,13,0.045,0.2')]
                                     )


def test_set_sweep_auto_abort(smu) -> None:
    mainframe = smu.parent

    smu.iv_sweep.sweep_auto_abort(constants.Abort.ENABLED)

    mainframe.write.assert_called_once_with("WM 2")


def test_get_sweep_auto_abort(smu) -> None:
    mainframe = smu.parent

    mainframe.ask.return_value = "WM2,2;WT1.0,0.0,0.0,0.0,0.0;"
    condition = smu.iv_sweep.sweep_auto_abort()
    assert condition == constants.Abort.ENABLED


def test_set_post_sweep_voltage_cond(smu) -> None:
    mainframe = smu.parent
    mainframe.ask.return_value = "WM2,2;WT1.0,0.0,0.0,0.0,0.0"
    smu.iv_sweep.post_sweep_voltage_condition(constants.WMDCV.Post.STOP)

    mainframe.write.assert_called_once_with("WM 2,2")


def test_get_post_sweep_voltage_cond(smu) -> None:
    mainframe = smu.parent

    mainframe.ask.return_value = "WM2,2;WT1.0,0.0,0.0,0.0,0.0"
    condition = smu.iv_sweep.post_sweep_voltage_condition()
    assert condition == constants.WM.Post.STOP
