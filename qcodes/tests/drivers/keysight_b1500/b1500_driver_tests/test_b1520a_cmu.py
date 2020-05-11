from unittest.mock import MagicMock, call

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
    # GroupParameter with initial values write at the init so reset the mock
    # to not count those write
    mainframe.reset_mock()

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


def test_ranging_mode(cmu):
    mainframe = cmu.parent

    cmu.ranging_mode(constants.RangingMode.AUTO)

    mainframe.write.assert_called_once_with('RC 3,0')


def test_sweep_auto_abort(cmu):
    mainframe = cmu.parent

    cmu.cv_sweep.sweep_auto_abort(constants.Abort.ENABLED)

    mainframe.write.assert_called_once_with("WMDCV 2")


def test_post_sweep_voltage_cond(cmu):
    mainframe = cmu.parent

    cmu.cv_sweep.post_sweep_voltage_cond(constants.WMDCV.Post.STOP)

    mainframe.write.assert_called_once_with("WMDCV 2,2")


def test_post_sweep_voltage_val_raise_warning_if_abort_not_set(cmu):
    pass


def test_cv_sweep_delay(cmu):
    mainframe = cmu.root_instrument
    from pprint import pprint

    cmu.cv_sweep.hold(1)
    cmu.cv_sweep.delay(1)
    #
    mainframe.write.assert_has_calls([call("WTDCV 1, 0, 0, 0, 0"),
                                      call("WTDCV 1, 1, 0, 0, 0")])


def test_cmu_sweep_steps(cmu):
    mainframe = cmu.root_instrument

    cmu.sweep_start(2)
    cmu.sweep_end(4)

    mainframe.write.assert_has_calls([call("WDCV 3, 1, 2, 0, 1"),
                                      call("WDCV 3, 1, 2, 4, 1")])


def test_setup_staircase_cv(cmu):
    cmu.setup_staircase_cv(
        v_start=-3,
        v_end=3,
        n_steps=201,
        freq=100e3,
        ac_rms=30e-3,
        post_sweep_voltage_cond=constants.WMDCV.Post.STOP,
        adc_mode=constants.ACT.Mode.PLC,
        adc_coef=5,
        imp_model=constants.IMP.MeasurementMode.Cp_D,
        ranging_mode=constants.RangingMode.AUTO,
        fixed_range_val=None,
        hold_delay=0,
        delay=0,
        step_delay=225e-3,
        trigger_delay=0,
        measure_delay=0,
        abort_enabled=constants.Abort.ENABLED,
        sweep_mode=constants.SweepMode.LINEAR,
        volt_monitor=False)

    assert cmu.adc_mode() == constants.ACT.Mode.PLC
    assert cmu.adc_coef() == 5
    assert cmu.frequency() == 100e3
    assert cmu.voltage_ac() == 30e-3
    assert cmu.cv_sweep.post_sweep_voltage_cond() == constants.WMDCV.Post.STOP
    assert cmu.cv_sweep.hold() == 0
    assert cmu.cv_sweep.delay() == 0
    assert cmu.cv_sweep.step_delay() == 225e-3
    assert cmu.cv_sweep.trigger_delay() == 0
    assert cmu.cv_sweep.measure_delay() == 0
    assert cmu.sweep_mode() == constants.SweepMode.LINEAR
    assert cmu.sweep_start() == -3
    assert cmu.sweep_end() == 3
    assert cmu.sweep_steps() == 201
    assert cmu.measurement_mode() == constants.MM.Mode.CV_DC_SWEEP
    assert cmu.impedance_model() == constants.IMP.MeasurementMode.Cp_D
    assert cmu.ranging_mode() == constants.RangingMode.AUTO
    assert cmu.measurement_range_for_non_auto() is None


def test_cv_sweep_measurement(cmu):
    mainframe = cmu.parent

    cmu.setup_fnc_already_run = True
    cmu.sweep_start(-3)
    cmu.sweep_end(3)
    cmu.sweep_steps(201)
    cmu.sweep_mode(constants.SweepMode.LINEAR)
    cmu.run_sweep()

    mainframe.ask.assert_called_once_with('XE')


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


def test_correction_get_reference_values(cmu):
    mainframe = cmu.parent

    mainframe.ask.return_value = '100,0.001,2'
    response = 'Mode: Cp_G, Primary Cp: 0.001 F, Secondary G: 2.0 S'
    assert response == cmu.correction.get_reference_values(
        constants.CalibrationType.OPEN)


def test_clear_and_set_default_frequency_list_for_correction(cmu):
    mainframe = cmu.parent

    cmu.correction.frequency_list.clear_and_set_default()

    mainframe.write.assert_called_once_with('CLCORR 3,2')


def test_clear_frequency_list_for_correction(cmu):
    mainframe = cmu.parent

    cmu.correction.frequency_list.clear()

    mainframe.write.assert_called_once_with('CLCORR 3,1')


def test_add_frequency_for_correction(cmu):
    mainframe = cmu.parent

    cmu.correction.frequency_list.add(1000)

    mainframe.write.assert_called_once_with('CORRL 3,1000')


def test_query_from_frequency_list_for_correction(cmu):
    mainframe = cmu.parent

    mainframe.ask.return_value = '25'

    assert pytest.approx(25) == cmu.correction.frequency_list.query()
    mainframe.ask.assert_called_once_with('CORRL? 3')


def test_query_at_index_from_frequency_list_for_correction(cmu):
    mainframe = cmu.parent

    mainframe.ask.return_value = '1234.567'

    assert pytest.approx(1234.567) == cmu.correction.frequency_list.query(
        index=0)
    mainframe.ask.assert_called_once_with('CORRL? 3,0')


def test_perform_correction(cmu):
    mainframe = cmu.parent

    mainframe.ask.return_value = 0

    response = cmu.correction.perform(
        constants.CalibrationType.OPEN)
    assert constants.CORR.Response.SUCCESSFUL == response


def test_perform_and_enable_correction(cmu):
    mainframe = cmu.parent

    mainframe.ask.side_effect = [
        '0',  # for correction status
        '1'   # for correction state (enabled/disabled)
    ]

    response = cmu.correction.perform_and_enable(
        constants.CalibrationType.OPEN)

    expected_response = f'Correction status ' \
                        f'{constants.CORR.Response.SUCCESSFUL.name} and ' \
                        f'Enable ' \
                        f'{constants.CORRST.Response.ON.name}'
    assert response == expected_response


def test_abort(cmu):
    mainframe = cmu.parent

    cmu.abort()

    mainframe.write.assert_called_once_with('AB')
