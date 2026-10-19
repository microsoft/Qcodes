import os
import pathlib
from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.mock_instruments import MockDAC, MockField, MockLockin
from qcodes.station import Station

if TYPE_CHECKING:
    from qcodes.instrument.delegate import InstrumentGroup

PARENT_DIR = pathlib.Path(__file__).parent.absolute()


@pytest.fixture(scope="function")
def dac() -> MockDAC:
    return MockDAC("dac", num_channels=3)


@pytest.fixture(scope="function")
def field_x() -> MockField:
    return MockField("field_x")


@pytest.fixture(scope="function")
def lockin() -> MockLockin:
    _lockin = MockLockin(name="lockin")
    return _lockin


@pytest.fixture(scope="function")
def station(dac: MockDAC, lockin: MockLockin, field_x: MockField) -> Station:
    _station = Station()
    _station.add_component(dac)
    _station.add_component(lockin)
    _station.add_component(field_x)
    return _station


@pytest.fixture()
def chip_config() -> str:
    return os.path.join(PARENT_DIR, "data/chip.yml")


@pytest.fixture()
def chip(station: Station, chip_config: str) -> "InstrumentGroup":
    if hasattr(station, "MockChip_123"):
        return station.MockChip_123

    station.load_config_file(chip_config)
    _chip = station.load_MockChip_123(station=station)
    return _chip


@pytest.fixture()
def chip_config_typo() -> str:
    return os.path.join(PARENT_DIR, "data/chip_typo.yml")
