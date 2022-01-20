import logging

import pytest

import qcodes.instrument.sims as sims
from qcodes.instrument_drivers.stahl import Stahl


@pytest.fixture(scope="function")
def stahl_instrument():
    visa_lib = sims.__file__.replace(
        '__init__.py',
        'stahl.yaml@sim'
    )

    inst = Stahl('Stahl', 'ASRL3', visalib=visa_lib)

    try:
        yield inst
    finally:
        inst.close()


def test_parse_idn_string():
    """
    Test that we can parse IDN strings correctly
    """
    assert Stahl.parse_idn_string("HV123 005 16 b") == {
        "model": "HV",
        "serial_number": "123",
        "voltage_range": 5.0,
        "n_channels": 16,
        "output_type": "bipolar"
    }

    with pytest.raises(
            RuntimeError,
            match="Unexpected instrument response"
    ):
        Stahl.parse_idn_string("HS123 005 16 bla b")


def test_get_idn(stahl_instrument):
    """
    Instrument attributes are set correctly after getting the IDN
    """
    assert stahl_instrument.IDN() == {
        "vendor": "Stahl",
        "model": "BS",
        "serial": "123",
        "firmware": None
    }

    assert stahl_instrument.n_channels == 16
    assert stahl_instrument.voltage_range == 5.0
    assert stahl_instrument.output_type == "bipolar"


def test_get_set_voltage(stahl_instrument, caplog):
    """
    Test that we can correctly get/set voltages
    """
    with caplog.at_level(logging.DEBUG, logger="qcodes.instrument.base"):
        stahl_instrument.channel[0].voltage(1.2)
    assert stahl_instrument.channel[0].voltage() == -1.2
    assert any("Querying" in rec.message for rec in caplog.records)
    assert any("Response" in rec.message for rec in caplog.records)
    assert all(
        "did not produce an acknowledge reply" not in rec.message
        for rec in caplog.records
    )


def test_get_set_voltage_assert_warning(stahl_instrument, caplog):
    """
    On channel 2 we have deliberately introduced an error in the
    visa simulation; setting a voltage does not produce an acknowledge
    string. Test that a warning is correctly issued.
    """
    with caplog.at_level(logging.DEBUG, logger="qcodes.instrument.base"):
        stahl_instrument.channel[1].voltage(1.0)
    assert any(
        "did not produce an acknowledge reply" in rec.message for rec in caplog.records
    )


def test_get_current(stahl_instrument):
    """
    Test that we can read currents and that the unit is in Ampere
    """
    assert stahl_instrument.channel[0].current() == 1E-6
    assert stahl_instrument.channel[0].current.unit == "A"


def test_get_temperature(stahl_instrument):
    """
    Due to limitations in pyvisa-sim, we cannot test this.
    Line 191 of pyvisa-sim/component.py  should read
    "return response.encode('latin-1')" for this to work.
    """
    pass
