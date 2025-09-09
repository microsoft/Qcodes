import numpy as np
import pytest

from qcodes.instrument_drivers.CopperMountain.M5065 import CopperMountainM5065


class DummyM5065:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        return 1e6


@pytest.fixture()
def vna():
    # try:
    #     instance = CopperMountainModelM5065(
    #         name="M5065", address="TCPIP0::localhost::hislip0::INSTR", terminator="\n"
    #     )
    # except (ValueError, VisaIOError):
    # Either there is no VISA lib installed or there was no real
    # instrument found at the specified address => use simulated instrument
    instance = CopperMountainM5065(
        name="M5065",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="CopperMountain_M5065.yaml",
    )

    instance.reset()

    yield instance

    instance.close()


def test_m5065_instantiation(vna):
    assert vna.name == "M5065"
    assert vna._address == "GPIB::1::INSTR"


def test_idn_command(vna):
    idn = vna.get_idn()
    assert idn["vendor"] == "CMT"
    assert idn["model"] == "M5065"


def test_output_param(vna):
    vna.output(True)
    assert vna.output() is True
    vna.output(False)
    assert vna.output() is False


def test_power_param(vna):
    vna.power(-20)
    assert vna.power() == -20


def test_if_bandwidth_param(vna):
    vna.if_bandwidth(1000)
    assert vna.if_bandwidth() == 1000


def test_averages_param(vna):
    vna.averages(2)
    assert vna.averages() == 2


def test_frequency_params(vna):
    vna.start(1e6)
    vna.stop(2e6)
    vna.center(1.5e6)
    vna.span(1e6)
    assert vna.start() == 1e6
    assert vna.stop() == 2e6
    assert vna.center() == 1.5e6
    assert vna.span() == 1e6


def test_setting_start_raises_error_when_larger_than_stop(vna):
    with pytest.raises(ValueError, match="Stop frequency"):
        vna.stop(1e6)
        vna.start(1e7)


def test_number_of_points_param(vna):
    vna.number_of_points(201)
    assert vna.number_of_points() == 201


def test_trigger_source_param(vna):
    vna.trigger_source("bus")
    assert vna.trigger_source() == "bus"


def test_get_s(vna):
    result = vna.get_s()
    assert isinstance(result, tuple)
    assert len(result) == 9
    assert all(isinstance(arr, np.ndarray) for arr in result)


def test_update_lin_traces(vna):
    vna.update_lin_traces()


def test_reset_averages(vna):
    vna.reset_averages()
    assert vna.ask("SENS1:AVER?") == "0\n"


def test_point_iq(vna):
    vna.number_of_points(2)
    vna.start(1e6)
    vna.stop(vna.start() + 1)

    param = vna.point_s11_iq
    i, q = param.get_raw()
    assert isinstance(i, float)
    assert isinstance(q, float)


def test_point_mag_phase(vna):
    vna.number_of_points(2)
    vna.start(1e6)
    vna.stop(vna.start() + 1)

    param = vna.point_s11
    mag, phase = param.get_raw()
    assert isinstance(mag, float)
    assert isinstance(phase, float)
