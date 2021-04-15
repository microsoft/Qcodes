from qcodes.tests.instrument_mocks import MockField
from unittest.mock import patch

def test_mock_dac(dac):
    assert dac.ch01.voltage() == 0.
    dac.ch01.voltage(1.)
    assert dac.ch01.voltage() == 1.


def test_mock_field_meta(station, field_x, chip_config):
    with patch.object(MockField, "set_field", wraps=field_x.set_field) as mock_set_field:
        mock_set_field.side_effect = None
        station.load_config_file(chip_config)
        field = station.load_field(station=station)
        assert field.ramp_rate() == 0.02
        field.X(1.0)
        mock_set_field.assert_called_once_with(1.0, block=False)
