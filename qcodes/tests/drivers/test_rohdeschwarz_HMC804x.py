import pytest

from qcodes.instrument_drivers.rohde_schwarz.private.HMC804x import _RohdeSchwarzHMC804x
import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'RSHMC804x.yaml@sim')


@pytest.fixture(scope='module')
def HMC8041():
    hmc8041 = _RohdeSchwarzHMC804x('hmc8041',
                                   address='GPIB::1::65535::INSTR',
                                   num_channels=1,
                                   visalib=visalib,
                                   terminator='\n')
    yield hmc8041

    hmc8041.close()


@pytest.fixture(scope='module')
def HMC8042():
    hmc8042 = _RohdeSchwarzHMC804x('hmc8042',
                                   address='GPIB::1::65535::INSTR',
                                   num_channels=2,
                                   visalib=visalib,
                                   terminator='\n')
    yield hmc8042

    hmc8042.close()


@pytest.fixture(scope='module')
def HMC8043():
    hmc8043 = _RohdeSchwarzHMC804x('hmc8043',
                                   address='GPIB::1::65535::INSTR',
                                   num_channels=3,
                                   visalib=visalib,
                                   terminator='\n')
    yield hmc8043

    hmc8043.close()


def test_init(HMC8041, HMC8042, HMC8043):

    for hmc in [HMC8041, HMC8042, HMC8043]:

        idn_dict = hmc.IDN()

        assert idn_dict['vendor'] == 'QCoDeS'
