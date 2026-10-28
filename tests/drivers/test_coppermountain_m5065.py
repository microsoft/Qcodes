from typing import TYPE_CHECKING

import numpy as np
import pytest

from qcodes.instrument_drivers.CopperMountain import CopperMountainM5065

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest_mock import MockerFixture


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


@pytest.mark.parametrize("value", [3.0, np.float64(3)], ids=["float", "numpy_float"])
def test_averages_accepts_whole_floats(
    vna: CopperMountainM5065, mocker: "MockerFixture", value: float
) -> None:
    # whole-number floats, e.g. from numpy, are accepted and sent as int
    write_raw = mocker.spy(vna, "write_raw")

    vna.averages(value)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

    write_raw.assert_called_once_with("SENS1:AVER:COUN 3")
    assert vna.averages() == 3


def test_averages_rejects_non_integer(
    vna: CopperMountainM5065, mocker: "MockerFixture"
) -> None:
    # previously 2.5 was silently truncated to 2 by set_parser=int
    write_raw = mocker.spy(vna, "write_raw")

    with pytest.raises(TypeError):
        vna.averages(2.5)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    write_raw.assert_not_called()


def test_averages_enabled_enables_averages_trigger(
    vna: CopperMountainM5065,
) -> None:
    assert vna.averages_trigger_enabled() is False
    vna.averages_enabled(True)
    assert vna.averages_enabled() is True
    assert vna.averages_trigger_enabled() is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [("ON", True), ("off", False), (1, True), (0, False)],
)
def test_averages_trigger_enabled_accepts_on_off_strings_and_ints(
    vna: CopperMountainM5065, value: str | int, expected: bool
) -> None:
    # while it does not type check, on/off strings and ints are accepted
    vna.averages_trigger_enabled(value)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    assert vna.averages_trigger_enabled() is expected


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
