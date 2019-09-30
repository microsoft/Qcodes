from unittest.mock import MagicMock

import pytest

from qcodes.instrument_drivers.Keysight.keysightb1500 import constants
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


def test_phase_compensation_mode(cmu):
    mainframe = cmu.parent

    cmu.phase_compensation_mode(constants.ADJ.Mode.MANUAL)

    mainframe.write.assert_called_once_with('ADJ 3,1')

    assert constants.ADJ.Mode.MANUAL == cmu.phase_compensation_mode()


def test_phase_compensation(cmu):
    mainframe = cmu.parent

    mainframe.ask.return_value = 0

    response = cmu.phase_compensation()

    mainframe.ask.assert_called_once_with('ADJ? 3')
    assert isinstance(response, constants.ADJQuery.Response)
    assert response == constants.ADJQuery.Response.PASSED


def test_phase_compensation_with_mode(cmu):
    mainframe = cmu.parent

    mainframe.ask.return_value = 0

    response = cmu.phase_compensation(constants.ADJQuery.Mode.USE_LAST)

    mainframe.ask.assert_called_once_with('ADJ? 3,0')
    assert isinstance(response, constants.ADJQuery.Response)
    assert response == constants.ADJQuery.Response.PASSED


def test_enable_correction(cmu):
    mainframe = cmu.parent

    cmu.correction.enable(constants.CalibrationType.OPEN)
    mainframe.write.assert_called_once_with('CORRST 3,1,1')

    mainframe.reset_mock()

    cmu.correction.enable(constants.CalibrationType.SHORT)
    mainframe.write.assert_called_once_with('CORRST 3,2,1')

    mainframe.reset_mock()

    cmu.correction.enable(constants.CalibrationType.LOAD)
    mainframe.write.assert_called_once_with('CORRST 3,3,1')


def test_disable_correction(cmu):
    mainframe = cmu.parent

    cmu.correction.disable(constants.CalibrationType.OPEN)
    mainframe.write.assert_called_once_with('CORRST 3,1,0')

    mainframe.reset_mock()

    cmu.correction.disable(constants.CalibrationType.SHORT)
    mainframe.write.assert_called_once_with('CORRST 3,2,0')

    mainframe.reset_mock()

    cmu.correction.disable(constants.CalibrationType.LOAD)
    mainframe.write.assert_called_once_with('CORRST 3,3,0')


def test_correction_is_enabled(cmu):
    mainframe = cmu.parent

    mainframe.ask.return_value = '1'

    response = cmu.correction.is_enabled(constants.CalibrationType.SHORT)
    assert response == constants.CORRST.Response.ON


def test_correction_set_reference_values(cmu):
    mainframe = cmu.parent

    cmu.correction.set_reference_values(
        constants.CalibrationType.OPEN,
        constants.DCORR.Mode.Cp_G,
        1,
        2)
    mainframe.write.assert_called_once_with('DCORR 3,1,100,1,2')


def test_get_reference_value_for_correction(cmu):
    mainframe = cmu.parent

    mainframe.ask.return_value = '100,1,2'
    response = 'Mode: Cp_G, Primary (Cp/Ls): 1 in F/H, Secondary (G/Rs): 2 ' \
               'in S/Ω'
    assert response == cmu.correction.get_reference_value_for_correction(
        constants.CalibrationType.OPEN)


def test_clear_frequency_for_correction(cmu):
    mainframe = cmu.parent

    cmu.correction.clear_frequency_for_correction(
        constants.CLCORR.Mode.CLEAR_AND_SET_DEFAULT_FREQ)

    mainframe.write.assert_called_once_with('CLCORR 3,2')


def test_add_frequency_for_correction(cmu):
    mainframe = cmu.parent

    cmu.correction.add_frequency_for_correction(1000)

    mainframe.write.assert_called_once_with('CORRL 3,1000')


def test_get_frequency_list_for_correction(cmu):
    mainframe = cmu.parent

    mainframe.ask.return_value = 1

    assert pytest.approx(1) == \
           cmu.correction.get_frequency_list_for_correction()


def test_perform_correction(cmu):
    mainframe = cmu.parent

    mainframe.ask.return_value = 0

    response = cmu.correction.perform_correction(
        constants.CalibrationType.OPEN)
    assert constants.CORR.Response.SUCCESSFUL == response


def test_abort(cmu):
    mainframe = cmu.parent

    cmu.abort()

    mainframe.write.assert_called_once_with('AB')
