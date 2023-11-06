from __future__ import annotations

import math

import numpy as np
import pytest

from qcodes.validators import Bool

from .conftest import AClass, a_func

BOOLS: list[bool | np.bool_] = [True, False, np.bool_(True), np.bool_(False)]
NOTBOOLS = [0, 1, 10, -1, 100, 1000000, int(-1e15), int(1e15),
            0.1, -0.1, 1.0, 3.5, -2.3e6, 5.5e15, 1.34e-10, -2.5e-5,
            math.pi, math.e, '', None, float("nan"), float("inf"),
            -float("inf"), '1', [], {}, [1, 2], {1: 1}, b'good',
            AClass, AClass(), a_func]


def test_bool() -> None:
    b = Bool()

    for v in BOOLS:
        b.validate(v)

    for vv in NOTBOOLS:
        with pytest.raises(TypeError):
            b.validate(vv)  # type: ignore[arg-type]

        assert repr(b) == '<Boolean>'


def test_valid_bool_values() -> None:
    val = Bool()
    for vval in val.valid_values:
        val.validate(vval)
