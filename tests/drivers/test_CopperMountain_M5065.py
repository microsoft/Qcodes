import pytest

from qcodes.instrument_drivers.CopperMountain import CopperMountainM5065


class DummyM5065:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        return 1e6


@pytest.fixture()
def vna():
    instance = CopperMountainM5065(
        name="M5065",
        address="TCPIP0::localhost::hislip0::INSTR",
        pyvisa_sim_file="CopperMountain_M5065.yaml",
    )
    instance.reset()
    yield instance
    instance.close()


def test_m5065_instantiation(vna):
    assert vna.name == "M5065"
    assert vna._address == "TCPIP0::localhost::hislip0::INSTR"


def test_idn_command(vna):
    idn = vna.get_idn()
    assert idn["vendor"] == "CMT"
    assert idn["model"] == "M5065"


def test_min_and_max_frequency_defaults(vna):
    assert vna.min_freq == 300e3
    assert vna.max_freq == 6.5e9
