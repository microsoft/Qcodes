import pytest

import qcodes.instrument.sims as sims
from qcodes.instrument_drivers.AimTTi import AimTTiPL601

visalib = sims.__file__.replace('__init__.py', 'AimTTi_PL601P.yaml@sim')


@pytest.fixture(scope='function')
def driver():
    driver = AimTTiPL601("AimTTi", address="GPIB::1::INSTR", visalib=visalib)

    yield driver
    driver.close()


def test_idn(driver):
    assert {'firmware': '3.05-4.06',
            'model': 'PL601-P',
            'serial': '514710',
            'vendor': 'THURLBY THANDAR'} == driver.IDN()
