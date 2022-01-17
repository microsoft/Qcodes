import pytest
from qcodes.utils.validators import Callable


def test_callable():
    c = Callable()

    def test_func():
        return True

    c.validate(test_func)
    test_int = 5
    with pytest.raises(TypeError):
        c.validate(test_int)


def test_valid_values():
    val = Callable()
    for vval in val.valid_values:
        val.validate(vval)
