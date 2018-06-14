from qcodes.instrument_drivers.oxford.MercuryiPS_VISA import MercuryiPS
import qcodes.instrument.sims as sims
import pytest

visalib = sims.__file__.replace('__init__.py', 'MercuryiPS.yaml@sim')


@pytest.fixture(scope='function')
def driver():

    mips = MercuryiPS('mips', address='GPIB::1::65535::INSTR',
                      visalib=visalib)
    yield mips
    mips.close()


def test_idn(driver):
    assert driver.IDN()['model'] == 'SIMULATED MERCURY iPS'


def test_simple_setting(driver):
    """
    Some very simple setting of parameters. Mainly just to
    sanity-check the pyvisa-sim setup.
    """
    assert driver.GRPX.field_target() == 0
    driver.GRPX.field_target(0.1)
    assert driver.GRPX.field_target() == 0.1
