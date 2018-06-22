import pytest

from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Tektronix_AWG5014C.yaml@sim')


@pytest.fixture(scope='function')
def awg():
    awg_sim = Tektronix_AWG5014('awg_sim',
                                address='GPIB0::1::65535::INSTR',
                                timeout=1,
                                terminator='\n',
                                visalib=visalib)
    yield awg_sim

    awg_sim.close()


def test_init_awg(awg):

    idn_dict = awg.IDN()

    assert idn_dict['vendor'] == 'QCoDeS'
