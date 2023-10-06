import pytest

from qcodes.instrument_drivers.AimTTi import AimTTiPL601


@pytest.fixture(scope="function", name="driver")
def _make_driver():
    driver = AimTTiPL601(
        "AimTTi", address="GPIB::1::INSTR", pyvisa_sim_file="AimTTi_PL601P.yaml"
    )

    yield driver
    driver.close()


def test_idn(driver) -> None:
    assert {'firmware': '3.05-4.06',
            'model': 'PL601-P',
            'serial': '514710',
            'vendor': 'THURLBY THANDAR'} == driver.IDN()
