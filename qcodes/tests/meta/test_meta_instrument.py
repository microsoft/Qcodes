import numpy as np
from unittest.mock import patch
from qcodes.tests.instrument_mocks import MockField


def test_mock_dac(dac):
    assert dac.ch01.voltage() == 0.
    dac.ch01.voltage(1.)
    assert dac.ch01.voltage() == 1.


def test_mock_field_meta(station, field_x, chip_config):
    with patch.object(MockField, "set_field", wraps=field_x.set_field) as mock_set_field:
        station.load_config_file(chip_config)
        field = station.load_field(station=station)

        assert field.ramp_rate() == 0.02
        field.X(0.001)
        mock_set_field.assert_called_once_with(0.001, block=False)

        # Test group meta parameters
        field_x.set_field(0.001)
        ramp = field.ramp_X()
        assert ramp.field == 0.001
        assert ramp.ramp_rate == 0.02

        field.ramp_X({"field": 0.0, "ramp_rate": 10.0})
        assert field.ramp_rate() == 10.0
        assert field.X() == 0.0
