import pytest

from qcodes.instrument_drivers.Keysight.KeysightAgilent_33XXX import WaveformGenerator_33XXX
import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Keysight_33xxx.yaml@sim')


@pytest.fixture(scope='module')
def driver():
    kw_sim = WaveformGenerator_33XXX('kw_sim',
                                      address='GPIB::1::65535::INSTR',
                                      visalib=visalib)
    yield kw_sim

    kw_sim.close()


def test_init(driver):

    idn_dict = driver.IDN()

    assert idn_dict['vendor'] == 'QCoDeS'

    assert driver.model == '33522B'
    assert driver.num_channels == 2


def test_sync(driver):

    assert driver.sync.output() == 'OFF'
    driver.sync.output('ON')
    assert driver.sync.output() == 'ON'

    assert driver.sync.source() == 1
    # driver.sync.source(2)
    # assert driver.sync.source() == 2
