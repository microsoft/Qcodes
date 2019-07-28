import pytest

import qcodes.instrument.sims as sims
from qcodes.instrument_drivers.rigol.DS1074Z import RigolDrivers

# path to the .yaml file containing the simulated instrument
visalib = sims.__file__.replace('__init__.py', 'Rigol_DS1074Z.yaml@sim')



@pytest.fixture(scope='function')
def driver():
    rigol = RigolDrivers('rigol',
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

def test

