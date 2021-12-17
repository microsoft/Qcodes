from unittest.mock import patch

from numpy.testing import assert_almost_equal

from qcodes.tests.instrument_mocks import MockField


def test_mock_dac(dac):
    assert dac.ch01.voltage() == 0.
    dac.ch01.voltage(1.)
    assert dac.ch01.voltage() == 1.


def test_mock_field_delegate(station, field_x, chip_config):
    with patch.object(
        MockField, "set_field", wraps=field_x.set_field
    ) as mock_set_field:
        station.load_config_file(chip_config)
        field = station.load_field(station=station)

        assert field.ramp_rate() == 0.02
        field.X(0.001)
        mock_set_field.assert_called_once_with(0.001, block=False)

        # Test group delegate parameters
        field_x.set_field(0.001)
        ramp = field.ramp_X()
        assert_almost_equal(ramp.field, 0.001)
        assert ramp.ramp_rate == 0.02

        field.ramp_X(dict(field=0.0, ramp_rate=10.0))
        assert field.ramp_rate() == 10.0
        assert_almost_equal(field.X(), 0.0)
        assert field.ramp_X_ramp_rate() == 10.0
        assert_almost_equal(field.ramp_X_field(), 0.0)


def test_delegate_channel_instrument(station, chip_config):
    station.load_config_file(chip_config)
    switch = station.load_switch(station=station)

    state = switch.state01()
    assert state.dac_output == "off"
    assert state.smc == "off"
    assert state.bus == "off"
    assert state.gnd == "off"

    switch.state01(dict(dac_output="on", smc="off", bus="off", gnd="off"))
    state = switch.state01()
    assert state.dac_output == "on"
    assert state.smc == "off"
    assert state.bus == "off"
    assert state.gnd == "off"
