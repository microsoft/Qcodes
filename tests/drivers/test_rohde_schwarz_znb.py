from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.rohde_schwarz import RohdeSchwarzZNB20

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(name="znb")
def _make_znb() -> "Generator[RohdeSchwarzZNB20, None, None]":
    driver = RohdeSchwarzZNB20(
        "znb20",
        address="TCPIP0::localhost::inst0::INSTR",
        pyvisa_sim_file="Rohde_Schwarz_ZNB20.yaml",
        terminator="\n",
        init_s_params=False,
        reset_channels=False,
    )
    driver.add_channel("S21")
    yield driver
    driver.close()


def test_bandwidth_default(znb: RohdeSchwarzZNB20) -> None:
    assert znb.channels[0].bandwidth() == 10e3


def test_bandwidth_non_integer(znb: RohdeSchwarzZNB20) -> None:
    # The IF bandwidth can be set in 1 - 1.5 - 2 - 3 - 5 - 7 steps from 1 Hz
    # (see the R&S ZNB user manual, [SENSe<Ch>:]BANDwidth[:RESolution]), so
    # 1.5 Hz is a valid value, which the instrument reports as a non-integer
    channel = znb.channels[0]
    channel.bandwidth(1.5)
    bandwidth = channel.bandwidth()
    assert bandwidth == 1.5
    assert isinstance(bandwidth, float)

    # the CW trace setpoints are derived from the bandwidth
    npts = channel.npts()
    setpoints = channel.trace_fixed_frequency.setpoints
    assert setpoints is not None
    assert setpoints[0][0][-1] == pytest.approx(npts / 1.5)
