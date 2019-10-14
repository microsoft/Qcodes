import pytest
import numpy as np
from collections import Counter

from qcodes.instrument_drivers.AimTTi.AimTTi_PL601P_channels import AimTTi

import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'AimTTi_PL601P.yaml@sim')


@pytest.fixture(scope='function')
def driver():
    driver = AimTTi('AimTTi', address='ASRL4::INSTR')

    yield driver
    driver.close()

def test_idn(driver):
    assert {'firmware': '3.05-4.06',
            'model': 'PL601-P',
            'serial': '514710',
            'vendor': 'THURLBY THANDAR'} == driver.IDN()
