import pytest

from qcodes.instrument_drivers.Keysight.KeysightAgilent_33XXX import WaveformGenerator_33XXX
import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Keysight_33xxx.yaml@sim')


@pytest.fixture(scope='module')
def driver():
    kw_sim = WaveformGenerator_33XXX('kw_sim',
                                      address='GPIB::1::INSTR',
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
    driver.sync.source(2)
    assert driver.sync.source() == 2
    driver.sync.source(1)
    driver.sync.output('OFF')

def test_channel(driver):
    assert driver.ch1.function_type() == 'SIN'
    driver.ch1.function_type('SQU')
    assert driver.ch1.function_type() == 'SQU'
    driver.ch1.function_type('SIN')


def test_burst(driver):
    assert driver.ch1.burst_ncycles() == 1
    driver.ch1.burst_ncycles(10)
    assert driver.ch1.burst_ncycles() == 10
    driver.ch1.burst_ncycles(1)
    # the following does not actually work because
    # val parser cannot handle INF being returned.
    # not clear if this is a bug or the instrument get
    # set to something else?
    # driver.ch1.burst_ncycles('INF')
    # assert driver.ch1.burst_ncycles() == 'INF'
