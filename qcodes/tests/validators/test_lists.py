import pytest
from qcodes.utils.validators import Ints, Lists


def test_type():
    l = Lists()
    v1 = ['a', 'b', 5]
    l.validate(v1)

    v2 = 234
    with pytest.raises(TypeError):
        l.validate(v2)


def test_elt_vals():
    l = Lists(Ints(max_value=10))
    v1 = [0, 1, 5]
    l.validate(v1)

    v2 = [0, 1, 11]
    with pytest.raises(ValueError):
        l.validate(v2)


def test_valid_values():
    val = Lists(Ints(max_value=10))
    for vval in val.valid_values:
        val.validate(vval)
