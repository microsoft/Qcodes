import pytest
from qcodes.instrument_drivers.test import DriverTestCase

@pytest.fixture
def simulated_uut_instance():
    from qcodes.instrument_drivers.Keysight.keysight_b220x import KeysightB220X
    import qcodes.instrument.sims as sims
    path_to_yaml = sims.__file__.replace('__init__.py', 'keysight_b220x.yaml')

    uut = KeysightB220X('switch_matrix',
                        address='GPIB::1::INSTR',
                        visalib=path_to_yaml+'@sim'
                        )

    yield uut

    uut.close()


class TestSimulatedKeysightB220X:
    def test_idn_command(self, simulated_uut_instance):
        assert "AGILENT" in simulated_uut_instance.IDN()['vendor']


class TestRealKeysightB220X(DriverTestCase):
    from qcodes.instrument_drivers.Keysight.keysight_b220x import KeysightB220X
    driver = KeysightB220X


    def test_idn_command(self):
        assert "AGILENT" in self.instrument.IDN()['vendor']

    def test_set_channel_config_mode(self):
        self.instrument.ask(":ROUT:FUNC NCON")
        status_byte = self.instrument.ask("*ESR?")
        assert 0 == status_byte
        self.instrument.ask(":ROUT:FUNC ACON")
        status_byte = self.instrument.ask("*ESR?")
        assert 0 == status_byte


    def test_get_channel_config_mode(self):
        raise NotImplementedError

    def test_set_connection_rule(self):
        raise NotImplementedError

    def test_get_connection_rule(self):
        raise NotImplementedError

    def test_set_connection_sequence(self):
        raise NotImplementedError

    def test_get_connection_sequence(self):
        raise NotImplementedError

    def test_set_channel_name(self):
        raise NotImplementedError

    def test_get_channel_name(self):
        raise NotImplementedError

    def test_set_input_name(self):
        raise NotImplementedError

    def test_get_input_name(self):
        raise NotImplementedError

    # Relay control commands
    def test_open_all_relays_of_card(self):
        raise NotImplementedError

    def test_open_relays_in_list(self):
        raise NotImplementedError

    def test_query_open_relays_in_list(self):
        raise NotImplementedError

    def test_close_all_relays_of_card(self):
        raise NotImplementedError

    def test_close_relays_in_list(self):
        raise NotImplementedError

    def test_close_open_relays_in_list(self):
        raise NotImplementedError

    # Bias Mode commands
    def test_bias_disable_card(self):
        raise NotImplementedError

    def test_bias_disable_channels_in_list(self):
        raise NotImplementedError

    def test_bias_enable_card(self):
        raise NotImplementedError

    def test_bias_enable_channels_in_list(self):
        raise NotImplementedError

    def test_query_bias_enabled_channels_in_list(self):
        raise NotImplementedError

    def test_set_bias_input_port(self):
        raise NotImplementedError

    def test_query_bias_input_port(self):
        raise NotImplementedError

    def test_set_bias_status_for_card(self):
        raise NotImplementedError

    def test_query_bias_status_for_card(self):
        raise NotImplementedError

    # Ground Mode commands
    def test_ground_disable_card(self):
        raise NotImplementedError

    def test_ground_disable_channels_in_list(self):
        raise NotImplementedError

    def test_ground_enable_card(self):
        raise NotImplementedError

    def test_ground_enable_channels_in_list(self):
        raise NotImplementedError

    def test_query_ground_enabled_channels_in_list(self):
        raise NotImplementedError

    def test_set_ground_input_port(self):
        raise NotImplementedError

    def test_query_ground_input_port(self):
        raise NotImplementedError

    def test_set_ground_status_for_card(self):
        raise NotImplementedError

    def test_query_ground_status_for_card(self):
        raise NotImplementedError

    def test_set_ground_enabled_input_ports(self):
        raise NotImplementedError

    def test_query_ground_enabled_input_ports(self):
        raise NotImplementedError

    # Couple Mode
    def test_set_input_couple_ports(self):
        raise NotImplementedError

    def test_query_input_couple_ports(self):
        raise NotImplementedError

    def test_detect_input_couple_ports(self):
        raise NotImplementedError

    def test_set_input_couple_port_status_for_card(self):
        raise NotImplementedError

    # System subsystem
    def test_set_beep_state(self):
        raise NotImplementedError

    def test_query_card_configuration(self):
        raise NotImplementedError(self)

    def test_query_card_description(self):
        raise NotImplementedError

    def test_set_card_to_power_on_state(self):
        raise NotImplementedError

    def test_query_card_model_and_revision(self):
        raise NotImplementedError

    def test_set_lcd_state(self):
        raise NotImplementedError

    def test_set_led_state(self):
        raise NotImplementedError

    def test_set_lcd_display_string(self):
        raise NotImplementedError

    def test_query_error(self):
        raise NotImplementedError

    def test_set_frontpanel_keylock(self):
        raise NotImplementedError

    def test_save_setup(self):
        raise NotImplementedError

    def test_set_comment_for_saved_setup(self):
        raise NotImplementedError

    def test_delete_setup(self):
        raise NotImplementedError

    def test_set_lightpen_state(self):
        raise NotImplementedError

    def query_scpi_version(self):
        raise NotImplementedError
