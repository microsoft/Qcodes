import pytest
import re
from functools import partial

from qcodes.instrument_drivers.stahl import Stahl
from qcodes.instrument_drivers.stahl.stahl import (
    UnexpectedInstrumentResponse, chain
)


def test_parse_simple():
    """
    Test that we can parse simple messages from the instrument
    and that the proper exception is raised when the received
    response does not match the expectation
    """
    parser = Stahl.regex_parser("^QCoDeS is (.*)$")
    result, = parser("QCoDeS is Cool")
    assert result == "Cool"

    with pytest.raises(UnexpectedInstrumentResponse):
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


def test_parse_idn_string():

    idn_reply = "BS123 005 16 b"
    assert Stahl.parse_idn_string(idn_reply) == {
        "model": "BS",
        "serial_number": "123",
        "voltage_range": 5.0,
        "n_channels": 16,
        "output_type": "bipolar"
    }
