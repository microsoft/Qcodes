import pytest

from qcodes.instrument_drivers.tektronix.Keithley_s46 import (
    S46, LockAcquisitionError
)

import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Keithley_s46.yaml@sim')


@pytest.fixture(scope='module')
def s46():
    driver = S46('s46_sim',address='GPIB::2::INSTR', visalib=visalib)
    yield driver
    driver.close()


def test_runtime_error_on_bad_init():

    with pytest.raises(
        RuntimeError,
        match="The driver is initialized from an undesirable instrument state"
    ):
        S46('s46_bad_state', address='GPIB::1::INSTR', visalib=visalib)


def test_init(s46):

    n_channels = len(s46.channels)
    assert n_channels == 26

    closed_channels = [0, 7, 12]

    for channel_nr in range(n_channels):
        assert s46.channels[channel_nr].state() == "close" \
            if channel_nr in closed_channels else "open"


def test_open_close(s46):

    with pytest.raises(
        LockAcquisitionError,
        match="Relay already in use by channel"
    ):
        s46.channels[1].state("close")

    s46.channels[0].state("open")
    s46.channels[1].state("close")
    s46.channels[18].state("close")

    with pytest.raises(
        LockAcquisitionError,
        match="Relay already in use by channel"
    ):
        s46.channels[19].state("close")

    s46.channels[18].state("open")
    s46.channels[19].state("close")
