import re
from unittest.mock import MagicMock

import pytest

from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1517A import \
    B1517A
from qcodes.instrument_drivers.Keysight.keysightb1500.constants import \
    VOutputRange, CompliancePolarityMode, IOutputRange, IMeasRange


@pytest.fixture
def mainframe():
    yield MagicMock()


@pytest.fixture
def smu(mainframe):
    slot_nr = 1
    smu = B1517A(parent=mainframe, name='B1517A', slot_nr=slot_nr)
    yield smu


def test_snapshot():
    from qcodes.instrument.base import InstrumentBase
    # We need to use `InstrumentBase` (not a bare mock) in order for
    # `snapshot` methods call resolution to work out
    mainframe = InstrumentBase(name='mainframe')
    mainframe.write = MagicMock()
    slot_nr = 1
    smu = B1517A(parent=mainframe, name='B1517A', slot_nr=slot_nr)

    smu.use_high_speed_adc()
    smu.source_config(output_range=VOutputRange.AUTO)
    smu.measure_config(measure_range=IMeasRange.AUTO)

    s = smu.snapshot()

    assert '_source_config' in s
    assert 'output_range' in s['_source_config']
    assert isinstance(s['_source_config']['output_range'], VOutputRange)
    assert '_measure_config' in s
    assert 'measure_range' in s['_measure_config']
    assert isinstance(s['_measure_config']['measure_range'], IMeasRange)


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


def test_unconfigured_source_defaults_to_autorange_v(smu):
    mainframe = smu.parent

    smu.voltage(10)

    mainframe.write.assert_called_once_with('DV 1,0,10')


def test_unconfigured_source_defaults_to_autorange_i(smu):
    mainframe = smu.parent

    smu.current(0.2)

    mainframe.write.assert_called_once_with('DI 1,0,0.2')


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


def test_measure_current(smu):
    mainframe = smu.parent
    mainframe.ask.return_value = "NAI+000.005E-06\r"
    assert pytest.approx(0.005e-6) == smu.current()


def test_measure_voltage(smu):
    mainframe = smu.parent
    mainframe.ask.return_value = "NAV+000.123E-06\r"
    assert pytest.approx(0.123e-6) == smu.voltage()


def test_some_voltage_sourcing_and_current_measurement(smu):
    mainframe = smu.parent

    smu.source_config(output_range=VOutputRange.MIN_0V5, compliance=1e-9)
    smu.measure_config(IMeasRange.FIX_100nA)

    mainframe.ask.return_value = "NAI+000.005E-09\r"

    smu.voltage(6)

    mainframe.write.assert_called_once_with('DV 1,5,6,1e-09')

    assert pytest.approx(0.005e-9) == smu.current()


def test_use_high_resolution_adc(smu):
    mainframe = smu.parent

    smu.use_high_resolution_adc()

    mainframe.write.assert_called_once_with('AAD 1,1')


def test_use_high_speed_adc(smu):
    mainframe = smu.parent

    smu.use_high_speed_adc()

    mainframe.write.assert_called_once_with('AAD 1,0')
