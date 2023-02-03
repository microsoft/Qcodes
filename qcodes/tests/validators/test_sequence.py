import pytest

from qcodes.validators import Ints, Sequence


def test_type() -> None:
    l = Sequence()
    v1 = ['a', 'b', 5]
    l.validate(v1)

    l.validate((1, 2, 3))

    v2 = 234
    with pytest.raises(TypeError):
        l.validate(v2)  # type: ignore[arg-type]


def test_elt_vals() -> None:
    l = Sequence(Ints(max_value=10))
    v1 = [0, 1, 5]
    l.validate(v1)

    v2 = [0, 1, 11]
    with pytest.raises(ValueError, match="11 is invalid: must be between"):
        l.validate(v2)


def test_valid_values() -> None:
    val = Sequence(Ints(max_value=10))
    for vval in val.valid_values:
        val.validate(vval)


def test_length() -> None:
    l = Sequence(length=3)
    v1 = [0, 1, 5]
    l.validate(v1)

    v2 = [0, 1, 3, 4]
    with pytest.raises(ValueError, match="has not length"):
        l.validate(v2)

    v3 = [3, 4]
    with pytest.raises(ValueError, match="has not length"):
        l.validate(v3)


def test_sorted() -> None:
    l = Sequence(length=3, require_sorted=True)

    v1 = [0, 1, 5]
    l.validate(v1)

    v2 = (1, 3, 5)
    l.validate(v2)

    v3 = (1, 5, 3)
    with pytest.raises(ValueError, match="is required to be sorted"):
        l.validate(v3)

    v4 = [1, 7, 2]
    with pytest.raises(ValueError, match="is required to be sorted"):
        l.validate(v4)
