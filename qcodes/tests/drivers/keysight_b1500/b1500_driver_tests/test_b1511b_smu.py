from unittest.mock import MagicMock
import re

import pytest

from qcodes.instrument_drivers.Keysight.keysightb1500.KeysightB1511B import \
    B1511B
from qcodes.instrument_drivers.Keysight.keysightb1500.constants import \
    IOutputRange, IMeasRange

# pylint: disable=redefined-outer-name


@pytest.fixture
def mainframe():
    yield MagicMock()


@pytest.fixture
def smu(mainframe):
    slot_nr = 1
    smu = B1511B(parent=mainframe, name='B1511B', slot_nr=slot_nr)
    yield smu


def test_force_invalid_current_output_range_when_asu_not_present(smu):
    msg = re.escape("Invalid Source Current Output Range")
    with pytest.raises(RuntimeError, match=msg):
        smu.asu_present = True
        smu.asu_present = False
        smu.source_config(IOutputRange.MIN_1pA)


def test_i_measure_range_config_raises_invalid_range_error_when_asu_not_present(
        smu):
    msg = re.escape("8 current measurement range")
    with pytest.raises(RuntimeError, match=msg):
        smu.asu_present = True
        smu.asu_present = False
        smu.i_measure_range_config(IMeasRange.MIN_1pA)
