import pytest

from qcodes.instrument_drivers.stahl import Stahl
from qcodes.instrument_drivers.stahl.stahl import UnexpectedInstrumentResponse, chain


def test_parse_simple():
    """
    Test that we can parse simple messages from the instrument
    and that the proper exception is raised when the received
    response does not match the expectation
    """
    parser = Stahl.regex_parser("^QCoDeS is (.*)$")
    result = parser("QCoDeS is Cool")
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

