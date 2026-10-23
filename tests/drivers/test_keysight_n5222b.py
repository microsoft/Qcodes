from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.Keysight import KeysightN5222B

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(name="pna")
def _make_pna() -> "Generator[KeysightN5222B, None, None]":
    driver = KeysightN5222B(
        "n5222b",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="Keysight_N5222B.yaml",
    )
    yield driver
    driver.close()


def test_active_trace(pna: KeysightN5222B) -> None:
    active_trace = pna.active_trace()
    assert active_trace == 1
    assert isinstance(active_trace, int)

    # while it does not type check, make sure a non-integer trace number is
    # rejected at runtime
    with pytest.raises(TypeError):
        pna.active_trace(1.5)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("name", ["averages", "points"])
def test_int_parameters(pna: KeysightN5222B, name: str) -> None:
    param = pna.parameters[name]
    param(10)
    value = param()
    assert value == 10
    assert isinstance(value, int)

    # these are counts so a non-integer value is rejected by the validator
    # before anything is sent to the instrument
    with pytest.raises(TypeError):
        param(2.5)
    assert param() == 10
