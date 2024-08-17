import pytest

from qcodes.instrument_drivers.Keysight import (
    Keysight33xxx,
)


@pytest.fixture(scope="function")
def driver():
    kw_sim = Keysight33xxx(
        "kw_sim", address="GPIB::1::INSTR", pyvisa_sim_file="Keysight_33xxx.yaml"
    )
    yield kw_sim

    kw_sim.close()


def test_init(driver) -> None:
    idn_dict = driver.IDN()

    assert idn_dict["vendor"] == "QCoDeS"

    assert driver.model == "33522B"
    assert driver.num_channels == 2


def test_sync(driver) -> None:
    assert driver.sync.output() == "OFF"
    driver.sync.output("ON")
    assert driver.sync.output() == "ON"

    assert driver.sync.source() == 1
    driver.sync.source(2)
    assert driver.sync.source() == 2
    driver.sync.source(1)
    driver.sync.output("OFF")


def test_channel(driver) -> None:
    assert driver.ch1.function_type() == "SIN"
    driver.ch1.function_type("SQU")
    assert driver.ch1.function_type() == "SQU"
    driver.ch1.function_type("SIN")


def test_burst(driver) -> None:
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
