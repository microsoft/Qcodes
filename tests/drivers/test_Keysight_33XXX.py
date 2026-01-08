from typing import TYPE_CHECKING, assert_type

import pytest

from qcodes.instrument_drivers.Keysight import (
    Keysight33xxxOutputChannel,
    Keysight33xxxSyncChannel,
    Keysight33511B,
    Keysight33512B,
    Keysight33522B,
    Keysight33611A,
    Keysight33622A,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="function")
def driver() -> "Generator[Keysight33522B, None, None]":
    kw_sim = Keysight33522B(
        "kw_sim", address="GPIB::1::INSTR", pyvisa_sim_file="Keysight_33xxx.yaml"
    )
    yield kw_sim

    kw_sim.close()


def test_init(driver: Keysight33522B) -> None:
    idn_dict = driver.IDN()

    assert idn_dict["vendor"] == "QCoDeS"

    assert driver.model == "33522B"
    assert driver.num_channels == 2


def test_sync(driver: Keysight33522B) -> None:
    assert driver.sync.output() == "OFF"
    driver.sync.output("ON")
    assert driver.sync.output() == "ON"

    assert driver.sync.source() == 1
    driver.sync.source(2)
    assert driver.sync.source() == 2
    driver.sync.source(1)
    driver.sync.output("OFF")


def test_channel(driver: Keysight33522B) -> None:
    assert_type(driver.ch1, Keysight33xxxOutputChannel)
    assert_type(driver.ch2, Keysight33xxxOutputChannel)
    assert_type(driver.sync, Keysight33xxxSyncChannel)
    assert driver.ch1.function_type() == "SIN"
    driver.ch1.function_type("SQU")
    assert driver.ch1.function_type() == "SQU"
    driver.ch1.function_type("SIN")


def test_burst(driver: Keysight33522B) -> None:
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


def test_wrong_model_warns(
    caplog: pytest.LogCaptureFixture, request: pytest.FixtureRequest
) -> None:
    request.addfinalizer(caplog.clear)
    request.addfinalizer(Keysight33511B.close_all)
    request.addfinalizer(Keysight33512B.close_all)
    request.addfinalizer(Keysight33611A.close_all)
    request.addfinalizer(Keysight33622A.close_all)

    _ = Keysight33511B(
        "kw_sim_33511b", address="GPIB::1::INSTR", pyvisa_sim_file="Keysight_33xxx.yaml"
    )
    _ = Keysight33512B(
        "kw_sim_33512b", address="GPIB::1::INSTR", pyvisa_sim_file="Keysight_33xxx.yaml"
    )
    _ = Keysight33611A(
        "kw_sim_33611a", address="GPIB::1::INSTR", pyvisa_sim_file="Keysight_33xxx.yaml"
    )
    _ = Keysight33622A(
        "kw_sim_33622a", address="GPIB::1::INSTR", pyvisa_sim_file="Keysight_33xxx.yaml"
    )

    warns = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warns) >= 4
    assert (
        sum(
            [
                "The driver class name " in record.msg
                and "does not match the detected model" in record.msg
                for record in warns
            ]
        )
        == 4
    )
