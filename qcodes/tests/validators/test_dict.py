import pytest
from qcodes.utils.validators import Dict


def test_dict():
    d = Dict()
    test_dict = {}
    d.validate(test_dict)
    test_int = 5
    with pytest.raises(TypeError):
        d.validate(test_int)


def test_valid_values():
    val = Dict()
    for vval in val.valid_values:
        val.validate(vval)
