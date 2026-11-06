import itertools
from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.Keithley.Keithley_3706A import Keithley3706A

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="function", name="driver")
def _make_driver() -> "Generator[Keithley3706A, None, None]":
    driver = Keithley3706A(
        "Keithley_3706A",
        address="GPIB::11::INSTR",
        pyvisa_sim_file="Keithley_3706A.yaml",
    )
    yield driver
    driver.close()


@pytest.fixture()
def channels_in_slot_one() -> list[str]:
    channels = []
    for i in range(1, 7):
        for j in range(int(f"1{i}01"), int(f"1{i}17")):
            channels.append(str(j))
    return channels


@pytest.fixture()
def channels_in_slot_two() -> list[str]:
    channels = []
    for i in range(1, 7):
        for j in range(int(f"2{i}01"), int(f"2{i}17")):
            channels.append(str(j))
    return channels


@pytest.fixture()
def channels_in_slot_three() -> list[str]:
    channels = []
    for i in range(1, 7):
        for j in range(int(f"3{i}01"), int(f"3{i}17")):
            channels.append(str(j))
    return channels


@pytest.fixture()
def matrix_channels() -> list[str]:
    slots = ["1", "2", "3"]
    rows = [["1", "2", "3", "4", "5", "6"]] * 3
    columns = [
        [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
        ]
    ] * 3
    m_channels = []
    for i, slot in enumerate(slots):
        for element in itertools.product(slot, rows[i], columns[i]):
            m_channels.append("".join(element))
    return m_channels


def test_idn(driver: Keithley3706A) -> None:
    assert {
        "firmware": "01.56a",
        "model": "3706A-SNFP",
        "serial": "04447336",
        "vendor": "KEITHLEY INSTRUMENTS INC.",
    } == driver.get_idn()


def test_switch_card_idn(driver: Keithley3706A) -> None:
    assert {
        "slot_no": "1",
        "model": "3730",
        "mtype": "6x16 High Density Matrix",
        "firmware": "01.40h",
        "serial": "4447332",
    } == driver.get_switch_cards()[0]


def test_installed_card_id(driver: Keithley3706A) -> None:
    assert ["1", "2", "3"] == driver._get_slot_ids()


def test_slot_names(driver: Keithley3706A) -> None:
    assert ["slot1", "slot2", "slot3"] == driver._get_slot_names()


def test_get_interlock_state(driver: Keithley3706A) -> None:
    dict_list = (
        {
            "slot_no": "1",
            "state": (
                "No card is installed or the installed card does not support interlocks"
            ),
        },
        {"slot_no": "2", "state": "Interlocks 1 and 2 are disengaged on the card"},
        {
            "slot_no": "3",
            "state": "Interlock 1 is engaged, interlock 2 (if it exists) is disengaged",
        },
    )
    assert dict_list == driver.get_interlock_state()


def test_get_number_of_rows(driver: Keithley3706A) -> None:
    assert [6, 6, 6] == driver._get_number_of_rows()


def test_get_number_of_columns(driver: Keithley3706A) -> None:
    assert [16, 16, 16] == driver._get_number_of_columns()


@pytest.mark.parametrize("val", [0, 1, 2])
def test_get_rows(driver: Keithley3706A, val: int) -> None:
    rows_in_a_slot = ["1", "2", "3", "4", "5", "6"]
    assert rows_in_a_slot == driver._get_rows()[val]


@pytest.mark.parametrize("val", [0, 1, 2])
def test_get_columns(driver: Keithley3706A, val: int) -> None:
    columns_in_a_slot = [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
    ]
    assert columns_in_a_slot == driver._get_columns()[val]


@pytest.mark.parametrize("val", [1, 2, 3])
def test_number_of_channels_by_slot(driver: Keithley3706A, val: int) -> None:
    assert 96 == len(driver.get_channels_by_slot(val))


def test_total_number_of_channels(driver: Keithley3706A) -> None:
    assert 3 * 96 == len(driver.get_channels())


def test_channels_in_slot_one(
    driver: Keithley3706A, channels_in_slot_one: list[str]
) -> None:
    assert channels_in_slot_one == driver.get_channels_by_slot(1)


def test_channels_in_slot_two(
    driver: Keithley3706A, channels_in_slot_two: list[str]
) -> None:
    assert channels_in_slot_two == driver.get_channels_by_slot(2)


def test_channels_in_slot_three(
    driver: Keithley3706A, channels_in_slot_three: list[str]
) -> None:
    assert channels_in_slot_three == driver.get_channels_by_slot(3)


def test_matrix_channels(driver: Keithley3706A, matrix_channels: list[str]) -> None:
    assert matrix_channels == driver.get_channels()


def test_analog_backplane_specifiers(driver: Keithley3706A) -> None:
    specifiers = [
        "1911",
        "1912",
        "1913",
        "1914",
        "1915",
        "1916",
        "2911",
        "2912",
        "2913",
        "2914",
        "2915",
        "2916",
        "3911",
        "3912",
        "3913",
        "3914",
        "3915",
        "3916",
    ]
    assert specifiers == driver.get_analog_backplane_specifiers()


@pytest.mark.parametrize("val", ["slot1", "allslots", "3111", "3912", "2103:2116"])
def test_validator_truth(driver: Keithley3706A, val: str) -> None:
    assert driver._validator(val) is True


@pytest.mark.parametrize(
    "val", ["slot12", "alslots", "5111", "912", "123213", "QCoDeS"]
)
def test_validator_falsehood(driver: Keithley3706A, val: str) -> None:
    assert driver._validator(val) is False
