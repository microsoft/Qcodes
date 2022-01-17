import math

import numpy as np
import pytest
from qcodes.utils.validators import Ints

from .conftest import AClass, a_func

ints = [0, 1, 10, -1, 100, 1000000, int(-1e15), int(1e15),
        # warning: True==1 and False==0 - we *could* prohibit these, using
        # isinstance(v, bool)
        True, False,
        # np scalars
        np.int64(3)]
not_ints = [0.1, -0.1, 1.0, 3.5, -2.3e6, 5.5e15, 1.34e-10, -2.5e-5,
            math.pi, math.e, '', None, float("nan"), float("inf"),
            -float("inf"), '1', [], {}, [1, 2], {1: 1}, b'good',
            AClass, AClass(), a_func]


def test_unlimited():
    n = Ints()
    n.validate(n.valid_values[0])

    for v in ints:
        n.validate(v)

    for v in not_ints:
        with pytest.raises(TypeError):
            n.validate(v)


def test_min():
    for min_val in ints:
        n = Ints(min_value=min_val)
        for v in ints:
            if v >= min_val:
                n.validate(v)
            else:
                with pytest.raises(ValueError):
                    n.validate(v)

    for v in not_ints:
        with pytest.raises(TypeError):
            n.validate(v)


def test_max():
    for max_val in ints:
        n = Ints(max_value=max_val)
        for v in ints:
            if v <= max_val:
                n.validate(v)
            else:
                with pytest.raises(ValueError):
                    n.validate(v)

    for v in not_ints:
        with pytest.raises(TypeError):
            n.validate(v)


def test_range():
    n = Ints(0, 10)

    for v in ints:
        if 0 <= v <= 10:
            n.validate(v)
        else:
            with pytest.raises(ValueError):
                n.validate(v)

    for v in not_ints:
        with pytest.raises(TypeError):
            n.validate(v)

    assert repr(n) == '<Ints 0<=v<=10>'
    assert n.is_numeric


def test_failed_numbers():
    with pytest.raises(TypeError):
        Ints(1, 2, 3)

    with pytest.raises(TypeError):
        Ints(1, 1)  # min >= max

    for val in not_ints:
        with pytest.raises((TypeError, OverflowError)):
            Ints(max_value=val)

        with pytest.raises((TypeError, OverflowError)):
            Ints(min_value=val)


def test_valid_values():
    val = Ints()
    for vval in val.valid_values:
        val.validate(vval)
