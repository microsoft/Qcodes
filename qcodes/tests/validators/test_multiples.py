from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np
import pytest

from qcodes.validators import Multiples

from .conftest import AClass, a_func

divisors = (3, 7, 11, 13)
not_divisors: tuple[
    int | float | str | None | bytes | list | dict | type[AClass] | AClass | Callable,
    ...,
] = (
    0,
    -1,
    -5,
    -1e15,
    0.1,
    -0.1,
    1.0,
    3.5,
    -2.3e6,
    5.5e15,
    1.34e-10,
    -2.5e-5,
    math.pi,
    math.e,
    "",
    None,
    float("nan"),
    float("inf"),
    -float("inf"),
    "1",
    [],
    {},
    [1, 2],
    {1: 1},
    b"good",
    AClass,
    AClass(),
    a_func,
)
multiples: list[int | np.integer | bool] = [
    0,
    1,
    10,
    -1,
    100,
    1000000,
    int(-1e15),
    int(1e15),
    # warning: True==1 and False==0 - we *could* prohibit these, using
    # isinstance(v, bool)
    True,
    False,
    # numpy scalars
    np.int64(2),
]
not_multiples: tuple[
    float
    | Literal[""]
    | None
    | str
    | list
    | dict
    | bytes
    | type[AClass]
    | AClass
    | Callable,
    ...,
] = (
    0.1,
    -0.1,
    1.0,
    3.5,
    -2.3e6,
    5.5e15,
    1.34e-10,
    -2.5e-5,
    math.pi,
    math.e,
    "",
    None,
    float("nan"),
    float("inf"),
    -float("inf"),
    "1",
    [],
    {},
    [1, 2],
    {1: 1},
    b"good",
    AClass,
    AClass(),
    a_func,
)


def test_divisors() -> None:
    for d in divisors:
        n = Multiples(divisor=d)
        for v in [d * e for e in multiples]:
            n.validate(v)

        for vv in multiples:
            if vv == 0:
                continue
            with pytest.raises(ValueError):
                n.validate(vv)

        for vvv in not_multiples:
            with pytest.raises(TypeError):
                n.validate(vvv)  # type:ignore[arg-type]

    for dd in not_divisors:
        with pytest.raises(TypeError):
            n = Multiples(divisor=dd)  # type:ignore[arg-type]

    n = Multiples(divisor=3, min_value=1, max_value=10)
    assert repr(n) == '<Ints 1<=v<=10, Multiples of 3>'


def test_valid_values() -> None:
    for d in divisors:
        n = Multiples(divisor=d)
        for num in n.valid_values:
            n.validate(num)
