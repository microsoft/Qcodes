from unittest.mock import MagicMock

import pytest

from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1520A import \
    B1520A


@pytest.fixture
def mainframe():
    yield MagicMock()


@pytest.fixture
def cmu(mainframe):
    slot_nr = 3
    cmu = B1520A(parent=mainframe, name='B1520A', slot_nr=slot_nr)
    yield cmu


def test_force_dc_voltage(cmu):
    mainframe = cmu.parent

    cmu.voltage_dc(10)

    mainframe.write.assert_called_once_with('DCV 3,10')


def test_force_ac_voltage(cmu):
    mainframe = cmu.parent

    cmu.voltage_ac(0.1)

    mainframe.write.assert_called_once_with('ACV 3,0.1')


def test_set_ac_frequency(cmu):
    mainframe = cmu.parent

    cmu.frequency(100e3)

    mainframe.write.assert_called_once_with('FC 3,100000.0')


def test_get_capacitance(cmu):
    mainframe = cmu.parent

    mainframe.ask.return_value = "NCC-1.45713E-06,NCY-3.05845E-03"

    assert pytest.approx((-1.45713E-06, -3.05845E-03)) == cmu.capacitance()


def test_raise_error_on_unsupported_result_format(cmu):
    mainframe = cmu.parent

    mainframe.ask.return_value = "NCR-1.1234E-03,NCX-4.5677E-03"

    with pytest.raises(ValueError):
        cmu.capacitance()
