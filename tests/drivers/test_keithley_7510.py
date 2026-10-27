from typing import TYPE_CHECKING

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from qcodes.instrument_drivers.Keithley import Keithley7510

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest_mock import MockerFixture


@pytest.fixture(scope="function")
def dmm_7510_driver() -> "Generator[Keithley7510, None, None]":
    inst = Keithley7510(
        "Keithley_7510_sim",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="keithley_7510.yaml",
    )

    try:
        yield inst
    finally:
        inst.close()


def test_get_idn(dmm_7510_driver: Keithley7510) -> None:
    assert dmm_7510_driver.IDN() == {
        "vendor": "KEITHLEY INSTRUMENTS",
        "model": "DMM7510",
        "serial": "01234567",
        "firmware": "1.2.3a",
    }


def test_change_sense_function(dmm_7510_driver: Keithley7510) -> None:
    """
    Measurement should be the same as the sense function, e.g., only voltage
    measurement is allowed when the sense function is "voltage".
    """
    assert dmm_7510_driver.sense.function() == "voltage"
    with pytest.raises(AttributeError, match="no attribute 'current'"):
        dmm_7510_driver.sense.current()
    dmm_7510_driver.sense.function("current")
    assert dmm_7510_driver.sense.function() == "current"


@settings(deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(st.sampled_from((0.1, 1, 10, 100, 1000)), st.floats(0.01, 10))
def test_set_range_and_nplc(
    dmm_7510_driver: Keithley7510, upper_limit: float, nplc: float
) -> None:
    """
    Test the ability of setting range and nplc value for sense function.
    "Voltage" is used as an example.
    """
    dmm_7510_driver.sense.function("voltage")
    dmm_7510_driver.sense.range(upper_limit)
    assert dmm_7510_driver.sense.range() == upper_limit
    dmm_7510_driver.sense.nplc(nplc)
    assert dmm_7510_driver.sense.nplc() == nplc


def test_user_delay_is_float(dmm_7510_driver: Keithley7510) -> None:
    sense = dmm_7510_driver.sense
    sense.user_number(1)
    assert sense.user_delay() == 0.0

    sense.user_delay(0.2)
    user_delay = sense.user_delay()
    assert isinstance(user_delay, float)
    assert user_delay == 0.2


def test_average_count_is_int(dmm_7510_driver: Keithley7510) -> None:
    sense = dmm_7510_driver.sense
    assert sense.average_count() == 10

    sense.average_count(25)
    average_count = sense.average_count()
    assert isinstance(average_count, int)
    assert average_count == 25


@pytest.mark.parametrize("value", [25.0, np.float64(25)], ids=["float", "numpy_float"])
def test_average_count_accepts_whole_floats(
    dmm_7510_driver: Keithley7510, mocker: "MockerFixture", value: float
) -> None:
    # whole-number floats, e.g. from numpy, are accepted and sent as int
    sense = dmm_7510_driver.sense
    write_raw = mocker.spy(dmm_7510_driver, "write_raw")

    sense.average_count(value)  # pyright: ignore[reportArgumentType]

    write_raw.assert_called_once_with(":SENSe:voltage:AVERage:COUNt 25")
    assert sense.average_count() == 25


def test_average_count_rejects_non_integer(
    dmm_7510_driver: Keithley7510, mocker: "MockerFixture"
) -> None:
    write_raw = mocker.spy(dmm_7510_driver, "write_raw")

    with pytest.raises(TypeError):
        dmm_7510_driver.sense.average_count(2.5)  # pyright: ignore[reportArgumentType]
    write_raw.assert_not_called()


def test_digitize_acq_rate_is_int(dmm_7510_driver: Keithley7510) -> None:
    acq_rate = dmm_7510_driver.digi_sense.acq_rate()
    assert isinstance(acq_rate, int)
    assert acq_rate == 1_000_000


def test_digitize_count_is_int(dmm_7510_driver: Keithley7510) -> None:
    digi_sense = dmm_7510_driver.digi_sense
    assert digi_sense.count() == 10_000

    digi_sense.count(500)
    count = digi_sense.count()
    assert isinstance(count, int)
    assert count == 500
