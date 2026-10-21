import pytest

from qcodes.instrument_drivers.CopperMountain import CopperMountainM5180


class DummyM5065:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        return 1e6


@pytest.fixture()
def vna():
    instance = CopperMountainM5180(
        name="M5180",
        address="TCPIP0::localhost::hislip0::INSTR",
        pyvisa_sim_file="CopperMountain_M5180.yaml",
    )
    instance.reset()
    yield instance
    instance.close()


def test_m5180_instantiation(vna):
    assert vna.name == "M5180"
    assert vna.address == "TCPIP0::localhost::hislip0::INSTR"


def test_idn_command(vna):
    idn = vna.get_idn()
    assert idn["vendor"] == "CMT"
    assert idn["model"] == "M5180"


def test_min_and_max_frequency_defaults(vna):
    assert vna.start.vals.min_value == 300e3
    assert vna.stop.vals.max_value == 18e9


def test_min_and_max_power_defaults(vna):
    assert vna.power.vals.min_value == -40
    assert vna.power.vals.max_value == 10


def test_number_of_points_max_default(vna):
    assert vna.number_of_points.vals.max_value == 200_001
