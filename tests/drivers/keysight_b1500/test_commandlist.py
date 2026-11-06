from typing import TYPE_CHECKING

import pytest

from qcodes.instrument_drivers.Keysight.keysightb1500.message_builder import CommandList

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def uut() -> "Generator[CommandList, None, None]":
    yield CommandList()


def test_append(uut: CommandList) -> None:
    uut.append("a")
    uut.append("b")

    assert ["a", "b"] == uut


def test_set_final(uut: CommandList) -> None:
    uut.append("a")

    uut.set_final()

    with pytest.raises(ValueError):
        uut.append("b")


def test_clear(uut: CommandList) -> None:
    uut.append("a")
    uut.set_final()

    uut.clear()
    uut.append("b")

    assert ["b"] == uut


def test_string_representation(uut: CommandList) -> None:
    uut.append("a")
    uut.append("b")

    assert "a;b" == str(uut)
