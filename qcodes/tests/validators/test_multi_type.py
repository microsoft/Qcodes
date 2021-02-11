import pytest
from qcodes.utils.validators import Enum, Ints, Lists, MultiType, Strings

from .conftest import AClass, a_func


def test_good():
    m = MultiType(Strings(2, 4), Ints(10, 1000))

    for v in [10, 11, 123, 1000, 'aa', 'mop', 'FRED']:
        m.validate(v)

    for v in [9, 1001, 'Q', 'Qcode', None, 100.0, b'nice', [], {},
              a_func, AClass, AClass(), True, False]:
        with pytest.raises(ValueError):
            m.validate(v)

    assert repr(m) == '<MultiType: Strings 2<=len<=4, Ints 10<=v<=1000>'

    assert m.is_numeric

    assert not MultiType(Strings(), Enum(1, 2)).is_numeric


def test_bad():
    for args in [[], [1], [Strings(), True]]:
        with pytest.raises(TypeError):
            MultiType(*args)


def test_valid_values():
    ms = [MultiType(Strings(2, 4), Ints(10, 32)),
          MultiType(Ints(), Lists(Ints())),
          MultiType(Ints(), MultiType(Lists(Ints()), Ints()))]
    for m in ms:
        for val in m.valid_values:
            m.validate(val)
