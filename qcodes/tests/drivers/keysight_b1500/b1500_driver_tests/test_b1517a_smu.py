from unittest.mock import MagicMock

import pytest

from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500 import \
    B1517A
from qcodes.instrument_drivers.Keysight.keysightb1500.constants import \
    VOutputRange, CompliancePolarityMode, IOutputRange


@pytest.fixture
def mainframe():
    yield MagicMock()


@pytest.fixture
def smu(mainframe):
    slot_nr = 1
    smu = B1517A(parent=mainframe, name='B1517A', slot_nr=slot_nr)
    yield smu


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
    with pytest.raises(TypeError):
        smu.current(0.1)

    smu.source_config(output_range=IOutputRange.AUTO)
    with pytest.raises(TypeError):
        smu.voltage(0.1)


def test_measure_current(smu):
    mainframe = smu.parent
    mainframe.ask.return_value = "NAI+000.005E-06\r"
    assert pytest.approx(0.005e-6) == smu.current()


def test_measure_voltage(smu):
    mainframe = smu.parent
    mainframe.ask.return_value = "NAV+000.123E-06\r"
    assert pytest.approx(0.123e-6) == smu.voltage()
