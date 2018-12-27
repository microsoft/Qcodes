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
