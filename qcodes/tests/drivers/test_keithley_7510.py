# pylint: disable=redefined-outer-name
import pytest
from hypothesis import given
import hypothesis.strategies as st
from qcodes.instrument_drivers.tektronix.keithley_7510 import Keithley7510
import qcodes.instrument.sims as sims

VISALIB = sims.__file__.replace('__init__.py', 'keithley_7510.yaml@sim')


@pytest.fixture(scope="module")
def dmm_7510_driver():
    inst = Keithley7510('Keithley_7510_sim',
                        address='GPIB::1::INSTR',
                        visalib=VISALIB)

    try:
        yield inst
    finally:
        inst.close()


def test_get_idn(dmm_7510_driver):
    assert dmm_7510_driver.IDN() == {
        'vendor': 'KEITHLEY INSTRUMENTS',
        'model': 'DMM7510',
        'serial': '01234567',
        'firmware': '1.2.3a'
    }


def test_change_sense_function(dmm_7510_driver):
    """
    Measurement should be the same as the sense function, e.g., only voltage
    measurement is allowed when the sense function is "voltage".
    """
    assert dmm_7510_driver.sense.function() == 'voltage'
    with pytest.raises(
        AttributeError,
        match="no attribute 'current'"
    ):
        dmm_7510_driver.sense.current()
    dmm_7510_driver.sense.function('current')
    assert dmm_7510_driver.sense.function() == 'current'


@given(
    st.sampled_from((0.1, 1, 10, 100, 1000)),
    st.floats(0.01, 10)
)
def test_set_range_and_nplc(dmm_7510_driver, upper_limit, nplc):
    """
    Test the ability of setting range and nplc value for sense function.
    "Voltage" is used as an example.
    """
    dmm_7510_driver.sense.function('voltage')
    dmm_7510_driver.sense.range(upper_limit)
    assert dmm_7510_driver.sense.range() == upper_limit
    dmm_7510_driver.sense.nplc(nplc)
    assert dmm_7510_driver.sense.nplc() == nplc
