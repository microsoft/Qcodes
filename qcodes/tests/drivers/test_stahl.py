import pytest
import re
from functools import partial
import logging
import io

from qcodes.instrument_drivers.stahl import Stahl
from qcodes.instrument_drivers.stahl.stahl import chain
import qcodes.instrument.sims as sims


@pytest.fixture(scope="function")
def stahl_instrument():
    visa_lib = sims.__file__.replace(
        '__init__.py',
        'stahl.yaml@sim'
    )

    inst = Stahl('Stahl', 'ASRL3', visalib=visa_lib)
    inst.log.setLevel(logging.DEBUG)
    iostream = io.StringIO()
    lh = logging.StreamHandler(iostream)
    inst.log.logger.addHandler(lh)

    try:
        yield inst
    finally:
        inst.close()


def test_parse_simple():
    """
    Test that we can parse simple messages from the instrument
    and that the proper exception is raised when the received
    response does not match the expectation
    """
    parser = Stahl.regex_parser("^QCoDeS is (.*)$")
    result, = parser("QCoDeS is Cool")
    assert result == "Cool"

    with pytest.raises(
            RuntimeError,
            match=r"Unexpected instrument response"
    ):
        parser("QCoDe is Cool")


def test_parse_tuple():
    """
    Test that we can extract multiple values from a string
    """
    parser = Stahl.regex_parser(r"^QCoDeS version is (\d),(\d)$")
    a, b = parser("QCoDeS version is 3,4")

    assert a == "3"
    assert b == "4"


def test_query_voltage_current():
    """
    Simulate Stahl weirdness when querying voltage or current. We do not simply
    get a string representing a floating point value which we can convert by
    casting to float. Instead, we need to use regular expressions. Additionally,
    when querying voltage/current a comma instead of a period is used as the
    decimal separator! Send an angry email to Stahl about this!
    """
    parser = chain(
        Stahl.regex_parser("channel 1 voltage: (.*)$"),
        partial(re.sub, ",", "."),
        float
    )

    answer = parser("channel 1 voltage: 2,3")
    assert answer == 2.3


def test_chain():
    """
    Test chaining of callables
    """
    def f(a, b):
        return int(a) + int(b)

    parser = chain(
        Stahl.regex_parser(r"^QCoDeS version is (\d),(\d)$"),
        f
    )

    answer = parser("QCoDeS version is 3,4")
    assert answer == 7


def test_parse_idn_string(stahl_instrument):
    """
    1) Assert that the correct IDN message is received and parsed
    2) Instrument attributes are set correctly
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


def test_get_set_voltage(stahl_instrument):
    """
    Test that we can correctly get/set voltages
    """
    stahl_instrument.channel[0].voltage(1.2)
    assert stahl_instrument.channel[0].voltage() == -1.2
    logger = stahl_instrument.log.logger
    log_messages = logger.handlers[0].stream.getvalue()
    assert "did not produce an acknowledge reply" not in log_messages


def test_get_set_voltage_assert_warning(stahl_instrument):
    """
    On channel 2 we have deliberately introduced an error in the
    visa simulation; setting a voltage does not produce an acknowledge
    string. Test that a warning is correctly issued.
    """
    stahl_instrument.channel[1].voltage(1.0)
    logger = stahl_instrument.log.logger
    log_messages = logger.handlers[0].stream.getvalue()
    assert "did not produce an acknowledge reply" in log_messages


def test_get_current(stahl_instrument):
    """
    Test that we can read currents and that the unit is in Ampere
    """
    assert stahl_instrument.channel[0].current() == 1E-6
    assert stahl_instrument.channel[0].current.unit == "A"


def test_get_temperature(stahl_instrument):
    """
    Due to limitations in pyvisa-sim, we cannot test this.
    Line 191 of pyvisa-sim\component.py  should read
    "return response.encode('latin-1')" for this to work.
    """
    pass


