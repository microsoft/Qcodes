import pytest

from qcodes.validators import Strings

from .conftest import AClass, a_func

long_string = "+".join(str(i) for i in range(100000))
danish = "\u00d8rsted F\u00e6lled"
chinese = "\u590f\u65e5\u7545\u9500\u699c\u5927\u724c\u7f8e"

strings = [
    "",
    "0",
    "10",
    "1.0e+10",
    "a",
    "Ja",
    "Artichokes!",
    danish,
    chinese,
    long_string,
]

not_strings = [
    0,
    1,
    1.0e10,
    bytes("", "utf8"),
    bytes(danish, "utf8"),
    bytes(chinese, "utf8"),
    [],
    [1, 2, 3],
    {},
    {"a": 1, "b": 2},
    True,
    False,
    None,
    AClass,
    AClass(),
    a_func,
]


def test_unlimited() -> None:
    s = Strings()

    for v in strings:
        s.validate(v)

    for vv in not_strings:
        with pytest.raises(TypeError):
            s.validate(vv)  # type: ignore[arg-type]

    assert repr(s) == "<Strings>"


def test_min() -> None:
    for min_len in [0, 1, 5, 10, 100]:
        s = Strings(min_length=min_len)
        for v in strings:
            if len(v) >= min_len:
                s.validate(v)
            else:
                with pytest.raises(ValueError):
                    s.validate(v)

        for vv in not_strings:
            with pytest.raises(TypeError):
                s.validate(vv)  # type: ignore[arg-type]

    s = Strings(min_length=100)
    assert repr(s) == "<Strings len>=100>"


def test_max() -> None:
    for max_len in [1, 5, 10, 100]:
        s = Strings(max_length=max_len)
        for v in strings:
            if len(v) <= max_len:
                s.validate(v)
            else:
                with pytest.raises(ValueError):
                    s.validate(v)

    s = Strings(max_length=100)
    for vv in not_strings:
        with pytest.raises(TypeError):
            s.validate(vv)  # type: ignore[arg-type]

    assert repr(s) == "<Strings len<=100>"


def test_range() -> None:
    s = Strings(1, 10)

    for v in strings:
        if 1 <= len(v) <= 10:
            s.validate(v)
        else:
            with pytest.raises(ValueError):
                s.validate(v)

    for vv in not_strings:
        with pytest.raises(TypeError):
            s.validate(vv)  # type: ignore[arg-type]

    assert repr(s) == "<Strings 1<=len<=10>"

    # single-valued range
    assert repr(Strings(10, 10)) == "<Strings len=10>"


def test_failed_strings() -> None:
    with pytest.raises(TypeError):
        Strings(1, 2, 3)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        Strings(10, 9)

    with pytest.raises(TypeError):
        Strings(max_length=0)

    with pytest.raises(TypeError):
        Strings(min_length=1e12)  # type: ignore[arg-type]

    for length in [-1, 3.5, "2", None]:
        with pytest.raises(TypeError):
            Strings(max_length=length)  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            Strings(min_length=length)  # type: ignore[arg-type]


def test_valid_values() -> None:
    val = Strings()
    for vval in val.valid_values:
        val.validate(vval)
