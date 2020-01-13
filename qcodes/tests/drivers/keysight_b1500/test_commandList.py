import pytest

from qcodes.instrument_drivers.Keysight.keysightb1500.message_builder import \
    CommandList


@pytest.fixture
def uut():
    yield CommandList()


def test_append(uut: CommandList):
    uut.append('a')
    uut.append('b')

    assert ['a', 'b'] == uut


def test_set_final(uut):
    uut.append('a')

    uut.set_final()

    with pytest.raises(ValueError):
        uut.append('b')


def test_clear(uut):
    uut.append('a')
    uut.set_final()

    uut.clear()
    uut.append('b')

    assert ['b'] == uut


def test_string_representation(uut):
    uut.append('a')
    uut.append('b')

    assert 'a;b' == str(uut)
