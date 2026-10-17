from typing import TYPE_CHECKING
from unittest.mock import MagicMock, PropertyMock

import pytest
from pytest import FixtureRequest

from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1500_base import (
    KeysightB1500,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(name="b1500")
def _make_b1500(request: FixtureRequest) -> "Generator[KeysightB1500, None, None]":
    request.addfinalizer(KeysightB1500.close_all)

    instance = KeysightB1500(
        "SPA", address="GPIB::1::INSTR", pyvisa_sim_file="keysight_b1500.yaml"
    )

    instance.get_status()
    instance.reset()

    yield instance


@pytest.fixture(name="mainframe")
def _make_mainframe() -> "Generator[MagicMock, None, None]":
    PropertyMock()
    mainframe = MagicMock()
    name_parts = PropertyMock(return_value=["mainframe"])
    type(mainframe).name_parts = name_parts
    short_name = PropertyMock(return_value="short_name")
    type(mainframe).short_name = short_name
    full_name = PropertyMock(return_value="mainframe")
    type(mainframe).full_name = full_name
    yield mainframe
