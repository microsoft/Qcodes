import pytest

from qcodes.instrument_drivers.tektronix.AWG70002A import AWG70002A
import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Tektronix_AWG70000A.yaml@sim')


@pytest.fixture(scope='function')
def awg2():
    awg2_sim = AWG70002A('awg2_sim',
                         address='GPIB0::2::65535::INSTR',
                         visalib=visalib)
    yield awg2_sim

    awg2_sim.close()


def test_init_awg2(awg2):

    idn_dict = awg2.IDN()

    assert idn_dict['vendor'] == 'QCoDeS'
