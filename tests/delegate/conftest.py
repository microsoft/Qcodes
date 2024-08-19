import os
import pathlib

import pytest

from qcodes.instrument_drivers.mock_instruments import MockDAC, MockField, MockLockin
from qcodes.station import Station

PARENT_DIR = pathlib.Path(__file__).parent.absolute()


@pytest.fixture(scope="function")
def dac():
    return MockDAC("dac", num_channels=3)


@pytest.fixture(scope="function")
def field_x():
    return MockField("field_x")


@pytest.fixture(scope="function")
def lockin():
    _lockin = MockLockin(name="lockin")
    return _lockin


@pytest.fixture(scope="function")
def station(dac, lockin, field_x):
    _station = Station()
    _station.add_component(dac)
    _station.add_component(lockin)
    _station.add_component(field_x)
    return _station


@pytest.fixture()
def chip_config():
    return os.path.join(PARENT_DIR, "data/chip.yml")


@pytest.fixture()
def chip(station, chip_config):
    if hasattr(station, "MockChip_123"):
        return station.MockChip_123

    station.load_config_file(chip_config)
    _chip = station.load_MockChip_123(station=station)
    return _chip


@pytest.fixture()
def chip_config_typo():
    return os.path.join(PARENT_DIR, "data/chip_typo.yml")
