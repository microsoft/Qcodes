import pytest

from qcodes.instrument_drivers.CopperMountain import CopperMountainV0602


class DummyM5065:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        return 1e6


@pytest.fixture()
def vna():
    instance = CopperMountainV0602(
        name="V0602",
        address="TCPIP0::localhost::hislip0::INSTR",
        pyvisa_sim_file="CopperMountain_V0602.yaml",
    )
    instance.reset()
    yield instance
    instance.close()


def test_m5065_instantiation(vna):
    assert vna.name == "V0602"
    assert vna.address == "TCPIP0::localhost::hislip0::INSTR"


def test_idn_command(vna):
    idn = vna.get_idn()
    assert idn["vendor"] == "CMT"
    assert idn["model"] == "V0602"


def test_reset_does_not_leave_pending_response(vna: CopperMountainV0602) -> None:
    vna.reset()
    assert vna.get_idn()["model"] == "V0602"
    assert vna.output() is False


def test_min_and_max_frequency_defaults(vna):
    assert vna.start.vals.min_value == 100e3
    assert vna.stop.vals.max_value == 6.5e9


def test_min_and_max_power_defaults(vna):
    assert vna.power.vals.min_value == -50.0
    assert vna.power.vals.max_value == 5.0


def test_number_of_points_max_default(vna):
    assert vna.number_of_points.vals.max_value == 500_001


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


@pytest.mark.parametrize("units", ["MET", "FEET", "INCH"])
def test_electrical_distance_units(vna: CopperMountainV0602, units: str) -> None:
    vna.electrical_distance_units(units)
    assert vna.electrical_distance_units() == units


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
    vna.stop(1e6)
    with pytest.raises(ValueError, match="Stop frequency"):
        vna.start(1e7)


def test_number_of_points_param(vna):
    vna.number_of_points(201)
    assert vna.number_of_points() == 201


def test_trigger_source_param(vna):
    vna.trigger_source("bus")
    assert vna.trigger_source() == "bus"


def test_update_lin_traces(vna):
    vna.update_lin_traces()
