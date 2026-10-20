from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.CopperMountain import CopperMountainM5065

if TYPE_CHECKING:
    from collections.abc import Generator


class DummyM5065:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def start(self) -> float:
        return 1e6


@pytest.fixture()
def vna() -> "Generator[CopperMountainM5065, None, None]":
    instance = CopperMountainM5065(
        name="M5065",
        address="TCPIP0::localhost::hislip0::INSTR",
        pyvisa_sim_file="CopperMountain_M5065.yaml",
    )
    instance.reset()
    yield instance
    instance.close()


def test_m5065_instantiation(vna: CopperMountainM5065) -> None:
    assert vna.name == "M5065"
    assert vna.address == "TCPIP0::localhost::hislip0::INSTR"


def test_idn_command(vna: CopperMountainM5065) -> None:
    idn = vna.get_idn()
    assert idn["vendor"] == "CMT"
    assert idn["model"] == "M5065"


def test_min_and_max_frequency_defaults(vna: CopperMountainM5065) -> None:
    assert vna.min_freq == 300e3
    assert vna.max_freq == 6.5e9


def test_output_param(vna: CopperMountainM5065) -> None:
    vna.output(True)
    assert vna.output() is True
    vna.output(False)
    assert vna.output() is False


def test_power_param(vna: CopperMountainM5065) -> None:
    vna.power(-20)
    assert vna.power() == -20


def test_if_bandwidth_param(vna: CopperMountainM5065) -> None:
    vna.if_bandwidth(1000)
    assert vna.if_bandwidth() == 1000


def test_averages_param(vna: CopperMountainM5065) -> None:
    vna.averages(2)
    assert vna.averages() == 2


def test_frequency_params(vna: CopperMountainM5065) -> None:
    vna.start(1e6)
    vna.stop(2e6)
    vna.center(1.5e6)
    vna.span(1e6)
    assert vna.start() == 1e6
    assert vna.stop() == 2e6
    assert vna.center() == 1.5e6
    assert vna.span() == 1e6


def test_setting_start_raises_error_when_larger_than_stop(
    vna: CopperMountainM5065,
) -> None:
    vna.stop(1e6)
    with pytest.raises(ValueError, match="Stop frequency"):
        vna.start(1e7)


def test_number_of_points_param(vna: CopperMountainM5065) -> None:
    vna.number_of_points(201)
    assert vna.number_of_points() == 201


def test_trigger_source_param(vna: CopperMountainM5065) -> None:
    vna.trigger_source("bus")
    assert vna.trigger_source() == "bus"


def test_update_lin_traces(vna: CopperMountainM5065) -> None:
    vna.update_lin_traces()
