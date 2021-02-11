import numpy as np
import pytest
from qcodes.utils.validators import PermissiveInts


def test_close_to_ints():
    validator = PermissiveInts()
    validator.validate(validator.valid_values[0])

    a = 0
    b = 10
    values = np.linspace(a, b, b - a + 1)
    for i in values:
        validator.validate(i)


def test_bad_values():
    validator = PermissiveInts(0, 10)
    validator.validate(validator.valid_values[0])

    a = 0
    b = 10
    values = np.linspace(a, b, b - a + 2)
    for j, i in enumerate(values):
        if j == 0 or j == 11:
            validator.validate(i)
        else:
            with pytest.raises(TypeError):
                validator.validate(i)


def test_valid_values():
    val = PermissiveInts()
    for vval in val.valid_values:
        val.validate(vval)
