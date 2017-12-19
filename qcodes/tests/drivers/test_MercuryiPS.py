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
