import os
import numpy as np

from qcodes import Measurement


def test_mock_dac(dac):
    assert dac.ch01.voltage() == 0.
    dac.ch01.voltage(1.)
    assert dac.ch01.voltage() == 1.

def test_device(station, chip_config, dac, lockin):
    assert os.path.exists(chip_config)
    station.load_config_file(chip_config)
    chip = station.load_MockChip_123(station=station)
    assert station.dac == dac
    assert chip.device1.gate.endpoints == (station.dac.ch01.voltage,)
    assert chip.device1.source.endpoints == (
        lockin.frequency,
        lockin.amplitude,
        lockin.phase,
        lockin.time_constant
    )
    assert chip.device1.drain.endpoints == (
        lockin.X,
        lockin.Y
    )


def test_device_meas(station, chip):
    meas = Measurement(station=station)
    device = chip.device1
    meas.register_parameter(device.gate)  # register the first independent parameter
    meas.register_parameter(device.drain, setpoints=(device.gate,))  # now register the dependent oone
    meas.write_period = 2

    with meas.run() as datasaver:
        for set_v in np.linspace(0, 1.5, 10):
            device.gate.set(set_v)
            get_a = device.drain.X.get()
            datasaver.add_result((device.gate, set_v), (device.drain, get_a))
            datasaver.flush_data_to_database()
        assert len(datasaver.dataset.to_pandas_dataframe_dict()["device1_drain"]) == 10
