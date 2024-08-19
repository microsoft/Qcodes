import pytest
from pyvisa.errors import VisaIOError

from qcodes.instrument_drivers.Keysight.keysight_b220x import KeysightB220X


@pytest.fixture
def uut():
    try:
        resource_name = "insert_Keysight_B2200_VISA_resource_name_here"
        instance = KeysightB220X("switch_matrix", address=resource_name)
    except (ValueError, VisaIOError):
        # Either there is no VISA lib installed or there was no real
        # instrument found at the specified address => use simulated instrument
        instance = KeysightB220X(
            "switch_matrix",
            address="GPIB::1::INSTR",
            pyvisa_sim_file="keysight_b220x.yaml",
        )

    instance.get_status()
    instance.clear_status()
    instance.reset()

    yield instance

    instance.close()


def test_idn_command(uut) -> None:
    assert "AGILENT" in uut.IDN()["vendor"]
    assert 0 == uut.get_status()


def test_connect(uut) -> None:
    uut.connect(2, 48)
    assert 0 == uut.get_status()


def test_connect_throws_at_invalid_channel_number(uut) -> None:
    with pytest.raises(ValueError):
        uut.connect(2, 49)
    with pytest.raises(ValueError):
        uut.connect(2, 0)
    with pytest.raises(ValueError):
        uut.connect(0, 10)
    with pytest.raises(ValueError):
        uut.connect(15, 10)


def test_connect_emits_warning_on_statusbyte_not_null(uut) -> None:
    # some tricks are used to trigger an instrument error both in the
    # simulation as well as in the real instrument:
    # 1. with gnd mode enabled, it is illegal to connect to input channel 12,
    #  which will raise an instrument error in the physical instrument,
    # but not in the simulated instrument.
    # 2. The simulated instrument only accepts pre-defined channels in the
    # connect command. here a channel is used that was not pre-defined which
    # will cause an instrument error in the simulated instrument, but not in
    # the physical instrument

    uut.gnd_mode(True)
    with pytest.warns(UserWarning):
        uut.connect(12, 33)

        # The simulated instrument does not reset the settings to default
        # values, so gnd mode is explicitly disabled here:
        uut.gnd_mode(False)


def test_disconnect_throws_at_invalid_channel_number(uut) -> None:
    with pytest.raises(ValueError):
        uut.disconnect(2, 49)
    with pytest.raises(ValueError):
        uut.disconnect(2, 0)
    with pytest.raises(ValueError):
        uut.disconnect(0, 10)
    with pytest.raises(ValueError):
        uut.disconnect(15, 10)


def test_connections(uut) -> None:
    uut.connect(2, 48)
    uut.connect(10, 12)
    assert {(2, 48), (10, 12)} == uut.connections()


def test_to_channel_list(uut) -> None:
    assert "(@00345,01109)" == uut.to_channel_list([(3, 45), (11, 9)])


def test_connect_paths(uut) -> None:
    uut.disconnect_all()
    uut.connect_paths([(3, 45), (11, 9)])
    assert 0 == uut.get_status()


def test_disconnect_paths(uut) -> None:
    uut.connect_paths([(3, 45), (11, 9)])
    uut.disconnect_paths([(3, 45), (11, 9)])
    assert 0 == uut.get_status()


def test_disconnect_all(uut) -> None:
    uut.connect(2, 48)
    uut.disconnect_all()
    assert 0 == uut.get_status()


def test_disconnect(uut) -> None:
    uut.connect(2, 48)
    assert 0 == uut.get_status()
    uut.disconnect(2, 48)
    assert 0 == uut.get_status()
    uut.disconnect(3, 22)
    assert 0 == uut.get_status()


@pytest.mark.filterwarnings("ignore:When going")
def test_connection_rule(uut) -> None:
    uut.connection_rule("single")
    assert 0 == uut.get_status()
    assert "single" == uut.connection_rule()
    assert 0 == uut.get_status()


def test_connection_rule_emits_warning_when_going_from_free_to_single(uut) -> None:
    uut.connection_rule("free")  # uut should already be in free mode after reset

    with pytest.warns(UserWarning):
        uut.connection_rule("single")


def test_connection_sequence(uut) -> None:
    assert "bbm" == uut.connection_sequence()
    assert 0 == uut.get_status()
    uut.connection_sequence("mbb")
    assert 0 == uut.get_status()
    assert "mbb" == uut.connection_sequence()


def test_bias_disable_all_outputs(uut) -> None:
    uut.bias_disable_all_outputs()
    assert 0 == uut.get_status()


def test_bias_disable_ouput(uut) -> None:
    uut.bias_disable_output(1)
    assert 0 == uut.get_status()


def test_bias_enable_all_outputs(uut) -> None:
    uut.bias_enable_all_outputs()
    assert 0 == uut.get_status()


def test_bias_enable_output(uut) -> None:
    uut.bias_enable_output(1)
    assert 0 == uut.get_status()


def test_bias_input_port(uut) -> None:
    assert 10 == uut.bias_input_port()
    uut.bias_input_port(9)
    assert 9 == uut.bias_input_port()
    assert 0 == uut.get_status()


def test_bias_mode(uut) -> None:
    uut.bias_mode(True)
    assert uut.bias_mode()
    assert 0 == uut.get_status()


def test_gnd_disable_all_outputs(uut) -> None:
    uut.gnd_disable_all_outputs()
    assert 0 == uut.get_status()


def test_gnd_disable_output(uut) -> None:
    uut.gnd_disable_output(1)
    assert 0 == uut.get_status()


def test_gnd_enable_all_outputs(uut) -> None:
    uut.gnd_enable_all_outputs()
    assert 0 == uut.get_status()


def test_gnd_enable_output(uut) -> None:
    uut.gnd_enable_output(1)
    assert 0 == uut.get_status()


def test_gnd_input_port(uut) -> None:
    assert 12 == uut.gnd_input_port()
    uut.gnd_input_port(5)
    assert 5 == uut.gnd_input_port()
    assert 0 == uut.get_status()


def test_gnd_mode(uut) -> None:
    assert not uut.gnd_mode()
    uut.gnd_mode(True)
    assert uut.gnd_mode()
    assert 0 == uut.get_status()


def test_unused_inputs(uut) -> None:
    uut.unused_inputs()
    assert 0 == uut.get_status()

    uut.unused_inputs([3])
    assert 0 == uut.get_status()
    assert [3] == uut.unused_inputs()

    uut.unused_inputs([5, 6, 7, 8])
    assert 0 == uut.get_status()
    assert [5, 6, 7, 8] == uut.unused_inputs()


def test_couple_mode(uut) -> None:
    assert not uut.couple_mode()
    uut.couple_mode(True)
    assert uut.couple_mode()
    assert 0 == uut.get_status()


def test_couple_ports(uut) -> None:
    assert not uut.couple_ports()
    assert 0 == uut.get_status()

    uut.couple_ports([1])
    assert 0 == uut.get_status()
    assert [1] == uut.couple_ports()

    uut.couple_ports([1, 3, 5])
    assert 0 == uut.get_status()
    assert [1, 3, 5] == uut.couple_ports()

    with pytest.raises(ValueError):
        uut.couple_ports([2])
    with pytest.raises(ValueError):
        uut.couple_ports([2, 3])


def test_couple_port_autodetect(uut) -> None:
    uut.couple_port_autodetect()
    assert 0 == uut.get_status()


def test_get_error(uut) -> None:
    uut.get_error()
    assert 0 == uut.get_status()


class TestParseChannelList:
    @staticmethod
    def test_parse_channel_list() -> None:
        channel_list = "(@10101,10202)"
        assert {(1, 1), (2, 2)} == KeysightB220X.parse_channel_list(channel_list)

    @staticmethod
    def test_all_combinations_zero_padded() -> None:
        import itertools

        cards = range(5)
        inputs = range(1, 15)
        outputs = range(1, 49)

        for card, in_port, out_port in itertools.product(cards, inputs, outputs):
            padded = f"{card:01d}{in_port:02d}{out_port:02d}"

            assert {(in_port, out_port)} == KeysightB220X.parse_channel_list(padded)

    @staticmethod
    def test_all_combinations_unpadded() -> None:
        import itertools

        cards = range(5)
        inputs = range(1, 15)
        outputs = range(1, 49)

        for card, in_port, out_port in itertools.product(cards, inputs, outputs):
            padded = f"{card:01d}{in_port:02d}{out_port:02d}"
            unpadded = str(int(padded))

            assert {(in_port, out_port)} == KeysightB220X.parse_channel_list(unpadded)
