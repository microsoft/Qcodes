from __future__ import annotations

import hypothesis.strategies as hst
import pytest
from hypothesis import given

from qcodes.utils.types import numpy_complex
from qcodes.validators import ComplexNumbers


@given(complex_val=hst.complex_numbers())
@pytest.mark.filterwarnings("ignore:overflow encountered in cast:RuntimeWarning")
def test_complex(complex_val: complex) -> None:
    # we are generating 128 b complex numbers with hypothesis
    # so we are likely to generate overflow warnings when converting
    # these to np complex 64. In this case that is harmless.
    n = ComplexNumbers()
    assert str(n) == '<Complex Number>'
    n.validate(complex_val)
    n.validate(complex(complex_val))

    for complex_type in numpy_complex:
        n.validate(complex_type(complex_val))


@given(val=hst.one_of(hst.floats(), hst.integers(), hst.characters()))
def test_complex_raises(val: float | int | str) -> None:

    n = ComplexNumbers()

    with pytest.raises(TypeError, match=r"is not complex;"):
        n.validate(val)  # type: ignore[arg-type]
