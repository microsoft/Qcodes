from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from qcodes.validators import Numbers

from .conftest import AClass, a_func

numbers: list[float | bool | np.integer | np.floating] = [
    0,
    1,
    -1,
    0.1,
    -0.1,
    100,
    1000000,
    1.0,
    3.5,
    -2.3e6,
    5.5e15,
    1.34e-10,
    -2.5e-5,
    math.pi,
    math.e,
    # warning: True==1 and False==0
    True,
    False,
    # warning: +/- inf are allowed if max & min are not specified!
    -float("inf"),
    float("inf"),
    # numpy scalars
    np.int64(36),
    np.float32(-1.123),
]
not_numbers: list[Any] = [
    "",
    None,
    "1",
    [],
    {},
    [1, 2],
    {1: 1},
    b"good",
    AClass,
    AClass(),
    a_func,
]


def test_unlimited() -> None:
    n = Numbers()

    for v in numbers:
        n.validate(v)

    for v in not_numbers:
        with pytest.raises(TypeError):
            n.validate(v)

    # special case - nan now raises a ValueError rather than a TypeError
    with pytest.raises(ValueError):
        n.validate(float("nan"))

    n.validate(n.valid_values[0])


def test_min() -> None:
    values: list[float] = [-1e20, -1, -0.1, 0, 0.1, 10]
    for min_val in values:
        n = Numbers(min_value=min_val)

        n.validate(n.valid_values[0])
        for v in numbers:
            if v >= min_val:
                n.validate(v)
            else:
                with pytest.raises(ValueError):
                    n.validate(v)


def test_min_raises() -> None:
    n = Numbers(min_value=10)

    for v in not_numbers:
        with pytest.raises(TypeError):
            n.validate(v)

    with pytest.raises(ValueError):
        n.validate(float("nan"))


def test_max() -> None:
    for max_val in [-1e20, -1, -0.1, 0, 0.1, 10]:
        n = Numbers(max_value=max_val)

        n.validate(n.valid_values[0])
        for v in numbers:
            if v <= max_val:
                n.validate(v)
            else:
                with pytest.raises(ValueError):
                    n.validate(v)


def test_max_raises() -> None:
    n = Numbers(max_value=10)

    for v in not_numbers:
        with pytest.raises(TypeError):
            n.validate(v)

    with pytest.raises(ValueError):
        n.validate(float("nan"))


def test_range() -> None:
    n = Numbers(0.1, 3.5)

    for v in numbers:
        if 0.1 <= v <= 3.5:
            n.validate(v)
        else:
            with pytest.raises(ValueError):
                n.validate(v)

    for v in not_numbers:
        with pytest.raises(TypeError):
            n.validate(v)

    with pytest.raises(ValueError):
        n.validate(float("nan"))

    assert repr(n) == "<Numbers 0.1<=v<=3.5>"


def test_failed_numbers() -> None:
    with pytest.raises(TypeError):
        Numbers(1, 2, 3)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        Numbers(1, 1)  # min >= max

    for val in not_numbers:
        with pytest.raises(TypeError):
            Numbers(max_value=val)

        with pytest.raises(TypeError):
            Numbers(min_value=val)


def test_valid_values() -> None:
    val = Numbers()
    for vval in val.valid_values:
        val.validate(vval)
