import logging

import pytest

from qcodes.instrument_drivers.rigol.DG4000 import Rigol_DG4000
import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'RigolDG4000.yaml@sim')


@pytest.fixture(scope='module')
def driver():
    kw_sim = Rigol_DG4000('rigol_sim',
                          address='GPIB::1::INSTR',
                          visalib=visalib)
    yield kw_sim

    kw_sim.close()


def test_init(driver):
    assert driver.IDN()['vendor'] == 'QCoDeS'

    # test that what used to be implemented via add_function is still
    # accessible after having been turned into bound methods
    for ch in ['ch1_', 'ch2_']:
        for fun_attr in ['custom', 'harmonic', 'noise', 'pulse', 'ramp',
                         'sinusoid', 'square', 'user']:
            getattr(driver, f'{ch}{fun_attr}')


def test_counter_att(driver):
    assert driver.counter_attenuation() == 1


def test_apply_custom(driver, caplog):
    """
    Test the function that applies a user-defined waveform
    """

    freq = 1000
    amp = 2
    offset = 0
    phase = 45

    logger = 'qcodes.instrument.base.com.visa'

    with caplog.at_level(logging.DEBUG, logger=logger):
        driver.ch1_custom(freq, amp, offset, phase)
        expected_mssg = (f'[{driver.name}({type(driver).__name__})] '
                         f'Writing: SOUR1:APPL:CUST {freq:.6e},{amp:.6e},'
                         f'{offset:.6e},{phase:.6e}')
        assert caplog.records[0].message == expected_mssg


def _call_function_and_check_logging(driver, caplog,
                                     cmd_name: str, scpi_string: str):
    """
    Helper function for test_function_with_no_args
    """

    logger = 'qcodes.instrument.base.com.visa'

    with caplog.at_level(logging.DEBUG, logger=logger):
        getattr(driver, cmd_name)()
        expected_mssg = (f'[{driver.name}({type(driver).__name__})] '
                         f'Writing: {scpi_string}')
        assert caplog.records[0].message == expected_mssg
        caplog.clear()


def test_functions_with_no_args(driver, caplog):
    """
    Test a handful functions with no arguments
    """
    cmd_to_scpi = {'auto_counter': 'COUN:AUTO',
                   'reset': '*RST',
                   'shutdown': 'SYST:SHUTDOWN',
                   'restart': 'SYST:RESTART',
                   'beep': 'SYST:BEEP',
                   'copy_config_to_ch1': 'SYST:CSC CH2,CH1',
                   'copy_config_to_ch2': 'SYST:CSC CH1,CH2',
                   'copy_waveform_to_ch1': 'SYST:CWC CH2,CH1',
                   'copy_waveform_to_ch2': 'SYST:CWC CH1,CH2',
                   'ch1_align_phase': 'SOUR1:PHAS:INIT',
                   'ch2_align_phase': 'SOUR2:PHAS:INIT'}

    for cmd, scpi in cmd_to_scpi.items():
        _call_function_and_check_logging(driver, caplog, cmd_name=cmd,
                                         scpi_string=scpi)