import pytest

from qcodes.validators import (
    Enum,
    Ints,
    Lists,
    MultiType,
    Numbers,
    PermissiveMultiples,
    Strings,
)

from .conftest import AClass, a_func


def test_good_or() -> None:
    # combiner == 'OR'
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


def test_good_and() -> None:
    # combiner == 'AND'
    m = MultiType(
        Numbers(min_value=2e-3, max_value=5e4),
        PermissiveMultiples(divisor=1e-3),
        combiner="AND",
    )

    for v in [10, 11, 123, 1000, 49999.0, 49999.1, 49999.9, 50000.0, 0.1, 0.01, 0.002]:
        m.validate(v)

    # in py True == 1, so valid in this construction
    for vv in [
        0,
        0.001,
        50000.1,
        "Q",
        "Qcode",
        None,
        -1,
        b"nice",
        [],
        {},
        a_func,
        AClass,
        AClass(),
        False,
    ]:
        with pytest.raises(ValueError):
            m.validate(vv)

    assert (
        repr(m)
        == "<MultiType: Numbers 0.002<=v<=50000.0, PermissiveMultiples, Multiples of 0.001 to within 1e-09>"
    )
    assert m.is_numeric
    assert not MultiType(Strings(), Enum(1, 2), combiner="AND").is_numeric


def test_bad() -> None:
    # combiner == 'OR'
    for args in [[], [1], [Strings(), True]]:
        with pytest.raises(TypeError):
            MultiType(*args)  # type: ignore[misc]
    # combiner == 'OR'
    for args in [[], [1], [Strings(), True]]:
        with pytest.raises(TypeError):
            MultiType(*args, combiner="OR")  # type: ignore[misc]
    # combiner == 'AND'
    for args in [[], [1], [Strings(), True]]:
        with pytest.raises(TypeError):
            MultiType(*args, combiner="AND")  # type: ignore[misc]


def test_valid_values() -> None:
    # combiner == 'OR'
    ms = [MultiType(Strings(2, 4), Ints(10, 32)),
          MultiType(Ints(), Lists(Ints())),
          MultiType(Ints(), MultiType(Lists(Ints()), Ints()))]
    for m in ms:
        for val in m.valid_values:
            m.validate(val)
