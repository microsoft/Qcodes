import pytest

from qcodes.validators import Callable


def test_callable() -> None:
    c = Callable()

    def test_func() -> bool:
        return True

    c.validate(test_func)
    test_int = 5
    with pytest.raises(TypeError):
        c.validate(test_int)  # type: ignore[arg-type]


def test_valid_values() -> None:
    val = Callable()
    for vval in val.valid_values:
        val.validate(vval)
