import pytest
from qcodes.utils.validators import Dict


def test_dict():
    d = Dict()
    my_dict = {}
    d.validate(my_dict)
    my_int = 5
    with pytest.raises(TypeError):
        d.validate(my_int)


def test_valid_values():
    val = Dict()
    for vval in val.valid_values:
        val.validate(vval)
