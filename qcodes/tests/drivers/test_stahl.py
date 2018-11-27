import pytest

from qcodes.instrument_drivers.stahl import Stahl
from qcodes.instrument_drivers.stahl.stahl import UnexpectedInstrumentResponse, chain


def test_parse_simple():
    parser = Stahl.regex_parser("^QCoDeS is (.*)$")
    result = parser("QCoDeS is Cool")
    assert result == "Cool"

    with pytest.raises(UnexpectedInstrumentResponse):
        parser("QCoDe is Cool")


def test_parse_tuple():
    parser = Stahl.regex_parser(r"^QCoDeS version is (\d),(\d)$")
    a, b = parser("QCoDeS version is 3,4")

    assert a == "3"
    assert b == "4"


def test_chain():

    def f(a, b):
        return int(a) + int(b)

    parser = chain(
        Stahl.regex_parser(r"^QCoDeS version is (\d),(\d)$"),
        f
    )

    answer = parser("QCoDeS version is 3,4")
    assert answer == 7

