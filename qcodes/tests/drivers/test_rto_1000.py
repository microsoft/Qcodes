import pytest

from qcodes.instrument_drivers.rohde_schwarz.RTO1000 import RTO1000
import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'RTO_1000.yaml@sim')


@pytest.fixture(scope='function')
def driver():
    rto_sim = RTO1000('rto_sim',
                      address='GPIB::1::65535::INSTR',
                      visalib=visalib,
                      model='RTO1044'
                      )
    yield rto_sim

    rto_sim.close()


def test_init(driver):

    idn_dict = driver.IDN()

    assert idn_dict['vendor'] == 'QCoDeS'
