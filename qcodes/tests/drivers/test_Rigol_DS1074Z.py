import pytest

import qcodes.instrument.sims as sims
from qcodes.instrument_drivers.rigol.DS1074Z import DS1074Z

# path to the .yaml file containing the simulated instrument
visalib = sims.__file__.replace('__init__.py', 'Rigol_DS1074Z.yaml@sim')


@pytest.fixture(scope='function')
def driver():
    rigol = DS1074Z('rigol',
                    address='GPIB::1::INSTR',
                    # This matches the address in the .yaml file
                    visalib=visalib
                    )

    yield rigol
    rigol.close()


def test_initialize(driver):
    """
    Test that simple initialisation works
    """
    idn_dict = driver.IDN()
    assert idn_dict['vendor'] == 'QCoDeS'


def test_gets_correct_waveform_xorigin(driver):
    assert driver.waveform_xorigin() == 0


def test_gets_correct_waveform_xincrem(driver):
    assert driver.waveform_xincrem() == 0.1


def test_sets_correct_waveform_npoints(driver):
    driver.waveform_npoints(1000)
    assert driver.waveform_npoints() == 1000


def test_gets_correct_waveform_yorigin(driver):
    assert driver.waveform_yorigin() == 0


def test_gets_correct_waveform_yincrem(driver):
    assert driver.waveform_yincrem() == 0.1


def test_gets_correct_waveform_yref(driver):
    assert driver.waveform_yref() == 0


def test_sets_correct_trigger_mode(driver):
    driver.trigger_mode('edge')
    assert driver.trigger_mode() == "edge"
    driver.trigger_mode('pattern')
    assert driver.trigger_mode() == "pattern"
    driver.trigger_mode('pulse')
    assert driver.trigger_mode() == "pulse"
    driver.trigger_mode('video')
    assert driver.trigger_mode() == "video"


def test_get_data_source(driver):
    driver.data_source('ch1')
    assert driver.data_source() == "ch1"
    driver.data_source('ch2')
    assert driver.data_source() == "ch2"
