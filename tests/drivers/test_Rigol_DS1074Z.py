import pytest

from qcodes.instrument_drivers.rigol import RigolDS1074Z


@pytest.fixture(scope='function')
def driver():
    rigol = RigolDS1074Z(
        "rigol",
        address="GPIB::1::INSTR",
        # This matches the address in the .yaml file
        pyvisa_sim_file="Rigol_DS1074Z.yaml",
    )

    yield rigol
    rigol.close()


def test_initialize(driver) -> None:
    """
    Test that simple initialisation works
    """
    idn_dict = driver.IDN()
    assert idn_dict['vendor'] == 'QCoDeS'


def test_gets_correct_waveform_xorigin(driver) -> None:
    assert driver.waveform_xorigin() == 0


def test_gets_correct_waveform_xincrem(driver) -> None:
    assert driver.waveform_xincrem() == 0.1


def test_sets_correct_waveform_npoints(driver) -> None:
    driver.waveform_npoints(1000)
    assert driver.waveform_npoints() == 1000


def test_gets_correct_waveform_yorigin(driver) -> None:
    assert driver.waveform_yorigin() == 0


def test_gets_correct_waveform_yincrem(driver) -> None:
    assert driver.waveform_yincrem() == 0.1


def test_gets_correct_waveform_yref(driver) -> None:
    assert driver.waveform_yref() == 0


def test_sets_correct_trigger_mode(driver) -> None:
    driver.trigger_mode('edge')
    assert driver.trigger_mode() == "edge"
    driver.trigger_mode('pattern')
    assert driver.trigger_mode() == "pattern"
    driver.trigger_mode('pulse')
    assert driver.trigger_mode() == "pulse"
    driver.trigger_mode('video')
    assert driver.trigger_mode() == "video"


def test_get_data_source(driver) -> None:
    driver.data_source('ch1')
    assert driver.data_source() == "ch1"
    driver.data_source('ch2')
    assert driver.data_source() == "ch2"
