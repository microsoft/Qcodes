from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.CopperMountain import CopperMountainM5180

if TYPE_CHECKING:
    from collections.abc import Generator


class DummyM5065:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def start(self) -> float:
        return 1e6


@pytest.fixture()
def vna() -> "Generator[CopperMountainM5180, None, None]":
    instance = CopperMountainM5180(
        name="M5180",
        address="TCPIP0::localhost::hislip0::INSTR",
        pyvisa_sim_file="CopperMountain_M5180.yaml",
    )
    instance.reset()
    yield instance
    instance.close()


def test_m5180_instantiation(vna: CopperMountainM5180) -> None:
    assert vna.name == "M5180"
    assert vna.address == "TCPIP0::localhost::hislip0::INSTR"


def test_idn_command(vna: CopperMountainM5180) -> None:
    idn = vna.get_idn()
    assert idn["vendor"] == "CMT"
    assert idn["model"] == "M5180"


def test_min_and_max_frequency_defaults(vna: CopperMountainM5180) -> None:
    assert vna.min_freq == 300e3
    assert vna.max_freq == 18e9
