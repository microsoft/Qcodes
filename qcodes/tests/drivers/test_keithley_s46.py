import pytest

from qcodes.instrument_drivers.tektronix.Keithley_s46 import (
    S46, LockAcquisitionError
)

import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Keithley_s46.yaml@sim')


class S46LoggedAsk(S46):
    def __init__(self, *args, **kwargs):
        self._ask_log = []
        super().__init__(*args, **kwargs)

    def ask(self, cmd: str):
        self._ask_log.append(cmd)
        return super().ask(cmd)


@pytest.fixture(scope='module')
def s46_six():
    driver = S46LoggedAsk('s46_six', address='GPIB::2::INSTR', visalib=visalib)
    yield driver
    driver.close()


@pytest.fixture(scope='module')
def s46_four():
    driver = S46LoggedAsk('s46_four', address='GPIB::3::INSTR', visalib=visalib)
    yield driver
    driver.close()


def test_runtime_error_on_bad_init():

    with pytest.raises(
        RuntimeError,
        match="The driver is initialized from an undesirable instrument state"
    ):
        S46('s46_bad_state', address='GPIB::1::INSTR', visalib=visalib)


def test_init_six(s46_six):
    assert s46_six._ask_log.count(":CLOS?") == 1

    n_channels = len(s46_six.channels)
    assert n_channels == 26

    closed_channels = [1, 8, 13]

    for channel in s46_six.channels:
        channel_nr = S46.aliases[channel.short_name]
        state = "close" if channel_nr in closed_channels else "open"
        assert channel.state() == state


def test_init_four(s46_four):
    assert s46_four._ask_log.count(":CLOS?") == 1

    n_channels = len(s46_four.channels)
    assert n_channels == 18

    closed_channels = [1, 8]

    for channel in s46_four.channels:
        channel_nr = S46.aliases[channel.short_name]
        state = "close" if channel_nr in closed_channels else "open"
        assert channel.state() == state


def test_open_close(s46_six):

    with pytest.raises(
        LockAcquisitionError,
        match="is already in use by channel"
    ):
        s46_six.channels[1].state("close")

    s46_six.channels[0].state("open")
    s46_six.channels[1].state("close")
    s46_six.channels[18].state("close")

    with pytest.raises(
        LockAcquisitionError,
        match="is already in use by channel"
    ):
        s46_six.channels[19].state("close")


def alias_to_channel_nr(alias):
    offset = dict(zip(["A", "B", "C", "D", "R"], range(0, 32, 6)))[alias[0]]
    return offset + int(alias[1:])


def test_aliases_six(s46_six):
    for channel in s46_six.channels:
        alias = channel.short_name
        assert getattr(s46_six, alias) is channel
        assert channel.channel_number == alias_to_channel_nr(alias)


def test_aliases_four(s46_four):
    for channel in s46_four.channels:
        alias = channel.short_name
        assert getattr(s46_four, alias) is channel
        assert channel.channel_number == alias_to_channel_nr(alias)
