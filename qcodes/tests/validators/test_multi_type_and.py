import pytest
from qcodes.utils.validators import Enum, MultiTypeAnd, Strings, Numbers, PermissiveMultiples

from .conftest import AClass, a_func


def test_good():
    m = MultiTypeAnd(Numbers(min_value=2e-3, max_value=5e4), PermissiveMultiples(divisor=1e-3))

    for v in [10, 11, 123, 1000, 49999.0, 49999.1, 49999.9, 50000.0, 0.1, 0.01, 0.002]:
        m.validate(v)

    # in py True == 1, so valid in this construction
    for v in [0, 0.001, 50000.1, 'Q', 'Qcode', None, -1, b'nice', [], {},
              a_func, AClass, AClass(), False]:
        with pytest.raises(ValueError):
            m.validate(v)

    assert repr(m) == '<MultiTypeAnd: Numbers 0.002<=v<=50000.0, PermissiveMultiples, Multiples of 0.001 to within 1e-09>'
    assert m.is_numeric
    assert not MultiTypeAnd(Strings(), Enum(1, 2)).is_numeric


def test_bad():
    for args in [[], [1], [Strings(), True]]:
        with pytest.raises(TypeError):
            MultiTypeAnd(*args)


def test_valid_values():
    # the concept of checking Validator.valid_values does not hold for the 'AND' case
    pass
    #ms = [MultiTypeAnd(Numbers(min_value=0, max_value=120), PermissiveMultiples(divisor=0.1)),
    #      MultiTypeAnd(Numbers(min_value=2e-3, max_value=5e4), PermissiveMultiples(divisor=1e-3)),
    #      MultiTypeAnd(Anything(), MultiTypeAnd(Numbers(), Ints()))]
    #for m in ms:
    #    for val in m.valid_values:
    #        m.validate(val)
