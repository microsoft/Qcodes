import os

import numpy as np
import pytest

from qcodes import Measurement
from qcodes.tests.instrument_mocks import DummyChannel, MockCustomChannel


def test_device(station, chip_config, dac, lockin):
    assert os.path.exists(chip_config)
    station.load_config_file(chip_config)
    chip = station.load_MockChip_123(station=station)
    assert station.dac == dac
    assert chip.device1.gate.source_parameters == (station.dac.ch01.voltage,)
    assert chip.device1.source.source_parameters == (
        lockin.frequency,
        lockin.amplitude,
        lockin.phase,
        lockin.time_constant
    )
    assert chip.device1.drain.source_parameters == (
        lockin.X,
        lockin.Y
    )


@pytest.mark.usefixtures("experiment")
def test_device_meas(station, chip):
    meas = Measurement(station=station)
    device = chip.device1
    meas.register_parameter(device.gate)
    meas.register_parameter(device.drain, setpoints=(device.gate,))
    device.gate.inter_delay = 0
    device.gate.step = 1
    with meas.run() as datasaver:
        for set_v in np.linspace(0, 1.5, 10):
            device.gate.set(set_v)
            get_a = device.drain_X.get()
            datasaver.add_result((device.gate, set_v), (device.drain, get_a))
            datasaver.flush_data_to_database()
        assert len(
            datasaver.dataset.to_pandas_dataframe_dict()["device1_drain"]) == 10


def test_device_with_channels(chip, station):
    device = chip.channel_device

    assert device.gate_1 == station.dac.ch01
    assert device.gate_1.voltage.post_delay == 0.01
    assert device.readout.source_parameters == (station.lockin.phase,)
    assert device.readout() == 1e-5

    station.dac.ch01.voltage(-0.134)
    assert device.gate_1.voltage() == -0.134
    device.gate_1.voltage(-0.01)
    assert station.dac.ch01.voltage() == -0.01

    device.readout.source_parameters[0](0.5)
    assert station.lockin.phase() == 0.5
    station.lockin.phase(30)
    assert device.readout.source_parameters[0]() == 30
    assert device.readout() == 30


def test_device_with_custom_channels(chip, station):
    device = chip.channel_device_custom

    assert device.gate_1._dac_channel == station.dac.ch01
    assert device.gate_1.current_valid_range() == [-0.5, 0]
    assert device.gate_1.parent == station.dac
    assert isinstance(device.gate_1, MockCustomChannel)

    assert device.fast_gate._channel == 'dac.ch02'
    assert isinstance(device.fast_gate, DummyChannel)


def test_chip_definition(chip_config_typo, station):
    station.load_config_file(chip_config_typo)
    with pytest.raises(KeyError):
        _ = station.load_MockChip_123(station=station)
