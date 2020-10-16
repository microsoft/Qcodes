from unittest.mock import MagicMock
import re

import pytest

from qcodes.instrument.base import InstrumentBase
from qcodes.instrument_drivers.Keysight.keysightb1500 import constants
from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1511B import \
    B1511B
from qcodes.instrument_drivers.Keysight.keysightb1500.constants import \
    VOutputRange, CompliancePolarityMode, IOutputRange, IMeasRange, \
    VMeasRange

# pylint: disable=redefined-outer-name


@pytest.fixture
def mainframe():
    yield MagicMock()


@pytest.fixture
def smu(mainframe):
    slot_nr = 1
    smu = B1511B(parent=mainframe, name='B1511B', slot_nr=slot_nr)
    yield smu


def test_snapshot():
    # We need to use `InstrumentBase` (not a bare mock) in order for
    # `snapshot` methods call resolution to work out
    mainframe = InstrumentBase(name='mainframe')
    mainframe.write = MagicMock()
    slot_nr = 1
    smu = B1511B(parent=mainframe, name='B1511B', slot_nr=slot_nr)

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


def test_force_invalid_voltage_output_range(smu):
    msg = re.escape("Invalid Source Voltage Output Range")
    with pytest.raises(RuntimeError, match=msg):
        smu.source_config(VOutputRange.MIN_1500V)


def test_force_invalid_current_output_range(smu):
    msg = re.escape("Invalid Source Current Output Range")
    with pytest.raises(RuntimeError, match=msg):
        smu.source_config(IOutputRange.MIN_20A)


def test_force_invalid_current_output_range_when_asu_not_present(smu):
    msg = re.escape("Invalid Source Current Output Range")
    with pytest.raises(RuntimeError, match=msg):
        smu.source_config(IOutputRange.MIN_1pA)


def test_force_voltage_with_autorange(smu):
    mainframe = smu.parent

    smu.source_config(output_range=VOutputRange.AUTO)
    smu.voltage(10)

    mainframe.write.assert_called_once_with('DV 1,0,10')


def test_force_voltage_autorange_and_compliance(smu):
    mainframe = smu.parent

    smu.source_config(output_range=VOutputRange.AUTO,
                      compliance=1e-6,
                      compl_polarity=CompliancePolarityMode.AUTO,
                      min_compliance_range=IOutputRange.MIN_10uA)
    smu.voltage(20)

    mainframe.write.assert_called_once_with('DV 1,0,20,1e-06,0,15')


def test_new_source_config_should_invalidate_old_source_config(smu):
    mainframe = smu.parent

    smu.source_config(output_range=VOutputRange.AUTO,
                      compliance=1e-6,
                      compl_polarity=CompliancePolarityMode.AUTO,
                      min_compliance_range=IOutputRange.MIN_10uA)

    smu.source_config(output_range=VOutputRange.AUTO)
    smu.voltage(20)

    mainframe.write.assert_called_once_with('DV 1,0,20')


def test_force_current_with_autorange(smu):
    mainframe = smu.parent

    smu.source_config(output_range=IOutputRange.AUTO)
    smu.current(0.1)

    mainframe.write.assert_called_once_with('DI 1,0,0.1')


def test_raise_warning_output_range_mismatches_output_command(smu):
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


def test_some_voltage_sourcing_and_current_measurement(smu):
    mainframe = smu.parent

    smu.source_config(output_range=VOutputRange.MIN_0V5, compliance=1e-9)
    smu.i_measure_range_config(IMeasRange.FIX_100nA)

    mainframe.ask.return_value = "NAI+000.005E-09\r"

    smu.voltage(6)

    mainframe.write.assert_called_once_with('DV 1,5,6,1e-09')

    assert pytest.approx(0.005e-9) == smu.current()

    assert smu.voltage.measurement_status is None
    assert smu.current.measurement_status == constants.MeasurementStatus.N


def test_i_measure_range_config_raises_invalid_range_error_when_asu_not_present(
        smu):
    msg = re.escape("8 current measurement range")
    with pytest.raises(RuntimeError, match=msg):
        smu.i_measure_range_config(IMeasRange.MIN_1pA)
