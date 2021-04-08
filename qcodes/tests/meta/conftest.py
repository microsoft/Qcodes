import pytest
import os
import pathlib

import qcodes as qc
from qcodes.tests.instrument_mocks import MockField, MockLockin, MockDAC


PARENT_DIR = pathlib.Path(__file__).parent.absolute()


@pytest.fixture(scope="session")
def dac():
    return MockDAC('dac', num_channels=3)


@pytest.fixture(scope="session")
def field():
    return MockField('field_X')


@pytest.fixture(scope="session")
def lockin(dac, field):
    _lockin = MockLockin(
        name='lockin',
        setter_param=dac.ch01.voltage,
        field=field.field
    )
    return _lockin


@pytest.fixture(scope="session")
def station(dac, lockin):
    _station = qc.Station()
    _station.add_component(dac)
    _station.add_component(lockin)
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
