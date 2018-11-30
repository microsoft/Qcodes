import pytest
from itertools import product

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
        state = "close" if channel_nr in closed_channels else "open"
        assert s46.channels[channel_nr].state() == state


def test_open_close(s46):

    with pytest.raises(
        LockAcquisitionError,
        match="is already in use by channel"
    ):
        s46.channels[1].state("close")

    s46.channels[0].state("open")
    s46.channels[1].state("close")
    s46.channels[18].state("close")

    with pytest.raises(
        LockAcquisitionError,
        match="is already in use by channel"
    ):
        s46.channels[19].state("close")


def test_aliases(s46):

    hex_aliases = [
        f"{a}{b}" for a, b in product(
            ["A", "B", "C", "D"],
            list(range(1, 7))
        )
    ]

    aliases = hex_aliases + [f"R{i}" for i in range(1, 9)]

    for channel in s46.channels:
        idx = channel.channel_number - 1
        alias = aliases[idx]
        assert getattr(s46, alias) is channel
