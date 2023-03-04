from unittest.mock import MagicMock, PropertyMock

import pytest
from pytest import FixtureRequest
from pyvisa import VisaIOError

from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500_base import (
    KeysightB1500,
)


@pytest.fixture(name="b1500")
def _make_b1500(request: FixtureRequest):
    request.addfinalizer(KeysightB1500.close_all)

    try:
        resource_name = "insert_Keysight_B2200_VISA_resource_name_here"
        instance = KeysightB1500("SPA", address=resource_name)
    except (ValueError, VisaIOError):
        # Either there is no VISA lib installed or there was no real
        # instrument found at the specified address => use simulated instrument
        instance = KeysightB1500(
            "SPA", address="GPIB::1::INSTR", pyvisa_sim_file="keysight_b1500.yaml"
        )

    instance.get_status()
    instance.reset()

    yield instance


@pytest.fixture(name="mainframe")
def _make_mainframe():
    PropertyMock()
    mainframe = MagicMock()
    name_parts = PropertyMock(return_value=["mainframe"])
    type(mainframe).name_parts = name_parts
    short_name = PropertyMock(return_value="short_name")
    type(mainframe).short_name = short_name
    full_name = PropertyMock(return_value="mainframe")
    type(mainframe).full_name = full_name
    yield mainframe
