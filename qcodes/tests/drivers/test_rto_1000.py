import pytest

from qcodes.instrument_drivers.rohde_schwarz.RTO1000 import RTO1000


@pytest.fixture(scope='function')
def driver():
    rto_sim = RTO1000(
        "rto_sim",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="RTO_1000.yaml",
        model="RTO1044",
    )
    yield rto_sim

    rto_sim.close()


def test_init(driver) -> None:

    idn_dict = driver.IDN()

    assert idn_dict['vendor'] == 'QCoDeS'


def test_trigger_source_level(driver) -> None:
    assert driver.trigger_source() == 'CH1'
    assert driver.trigger_level() == 0
    driver.trigger_level(1.0)
    assert driver.trigger_level() == 1
    driver.trigger_level(0)
