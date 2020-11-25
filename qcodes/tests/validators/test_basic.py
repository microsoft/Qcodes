from typing import Any

import pytest
from qcodes.utils.validators import Anything, Validator

from .conftest import AClass, a_func


class BrokenValidator(Validator[Any]):

    def __init__(self):
        pass


def test_broken():
    # nor can you call validate without overriding it in a subclass
    b = BrokenValidator()
    with pytest.raises(NotImplementedError):
        b.validate(0)


def test_real_anything():
    a = Anything()
    for v in [None, 0, 1, 0.0, 1.2, '', 'hi!', [1, 2, 3], [],
              {'a': 1, 'b': 2}, {}, {1, 2, 3}, a, range(10),
              True, False, float("nan"), float("inf"), b'good',
              AClass, AClass(), a_func]:
        a.validate(v)

    assert repr(a) == '<Anything>'


def test_failed_anything():
    with pytest.raises(TypeError):
        Anything(1)

    with pytest.raises(TypeError):
        Anything(values=[1, 2, 3])
