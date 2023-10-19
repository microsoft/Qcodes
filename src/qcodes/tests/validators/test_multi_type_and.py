import pytest

from qcodes.validators import Enum, MultiTypeAnd, Numbers, PermissiveMultiples, Strings

from .conftest import AClass, a_func


def test_good() -> None:
    m = MultiTypeAnd(
        Numbers(min_value=2e-3, max_value=5e4), PermissiveMultiples(divisor=1e-3)
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
        repr(m) == "<MultiTypeAnd: Numbers 0.002<=v<=50000.0, "
        "PermissiveMultiples, Multiples of 0.001 to within 1e-09>"
    )
    assert m.is_numeric
    assert not MultiTypeAnd(Strings(), Enum(1, 2)).is_numeric


def test_bad() -> None:
    for args in ([], [1], [Strings(), True]):
        with pytest.raises(TypeError):
            MultiTypeAnd(*args)  # type: ignore[misc]
