import pytest
import qcodes.instrument_drivers.Keysight.Keysight_N6705B as N6705B

import qcodes.instrument.sims as sims

visalib = sims.__file__.replace('__init__.py', 'Keysight_N6705B.yaml@sim')


@pytest.fixture(scope='module')
def driver():
    driver = N6705B.N6705B('N6705B',
                           address="GPIB::1::INSTR",
                           visalib=visalib)
    yield driver
    driver.close()


def test_idn(driver):
    assert {'firmware': 'D.01.08',
            'model': 'N6705B',
            'serial': 'MY50001897',
            'vendor': 'Agilent Technologies'} == driver.IDN()


def test_channels(driver):
    # Ensure each channel got instantiated
    assert len(driver.channels) == 4
    for (i, ch) in enumerate(driver.channels):
        assert(ch.channel == i+1)
        assert(ch.ch_name == f"ch{i+1}")
