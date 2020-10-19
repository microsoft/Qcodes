import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import given
from qcodes.utils.validators import ComplexNumbers


@given(complex_val=hst.complex_numbers())
def test_complex(complex_val):

    n = ComplexNumbers()
    assert str(n) == '<Complex Number>'
    n.validate(complex_val)
    n.validate(np.complex(complex_val))
    n.validate(np.complex64(complex_val))
    n.validate(np.complex128(complex_val))


@given(val=hst.one_of(hst.floats(), hst.integers(), hst.characters()))
def test_complex_raises(val):

    n = ComplexNumbers()

    with pytest.raises(TypeError, match=r"is not complex;"):
        n.validate(val)
