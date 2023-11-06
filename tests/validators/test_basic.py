from __future__ import annotations

from typing import Any

import pytest

from qcodes.validators import Anything, Validator

from .conftest import AClass, a_func


class BrokenValidator(Validator[Any]):

    def __init__(self) -> None:
        pass


def test_broken() -> None:
    # nor can you call validate without overriding it in a subclass
    b = BrokenValidator()
    with pytest.raises(NotImplementedError):
        b.validate(0)


def test_real_anything() -> None:
    a = Anything()
    anything: list[Any] = [
        None,
        0,
        1,
        0.0,
        1.2,
        "",
        "hi!",
        [1, 2, 3],
        [],
        {"a": 1, "b": 2},
        {},
        {1, 2, 3},
        a,
        range(10),
        True,
        False,
        float("nan"),
        float("inf"),
        b"good",
        AClass,
        AClass(),
        a_func,
    ]
    for v in anything:
        a.validate(v)

    assert repr(a) == '<Anything>'


def test_failed_anything() -> None:
    with pytest.raises(TypeError):
        Anything(1)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        Anything(values=[1, 2, 3])  # type: ignore[call-arg]
