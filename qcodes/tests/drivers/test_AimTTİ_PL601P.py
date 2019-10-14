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


@pytest.fixture(scope='function')
def channels(driver):
    ch_names = {'ch1'}
    assert ch_names == set(list(driver.submodules.keys()))

    yield tuple(getattr(driver, ch_name)
                for ch_name in ch_names)


def test_idn(driver):
    assert {'firmware': '3.05-4.06',
            'model': 'PL601-P',
            'serial': '514710',
            'vendor': 'THURLBY THANDAR'} == driver.IDN()


def test_channel_parameters(driver):
    driver.ch1.volt(1.0)
    assert 1.0 == driver.ch1.volt()
