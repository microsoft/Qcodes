import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import given

from qcodes.utils.validators import ComplexNumbers
from qcodes.utils.types import numpy_complex


@given(complex_val=hst.complex_numbers())
def test_complex(complex_val):

    n = ComplexNumbers()
    assert str(n) == '<Complex Number>'
    n.validate(complex_val)
    n.validate(complex(complex_val))

    for complex_type in numpy_complex:
        n.validate(complex_type(complex_val))


@given(val=hst.one_of(hst.floats(), hst.integers(), hst.characters()))
def test_complex_raises(val):

    n = ComplexNumbers()

    with pytest.raises(TypeError, match=r"is not complex;"):
        n.validate(val)
