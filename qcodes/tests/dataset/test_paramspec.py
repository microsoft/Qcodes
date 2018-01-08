import pytest
from numbers import Number

from numpy import ndarray
from hypothesis import given
import hypothesis.strategies as hst

from qcodes.dataset.param_spec import ParamSpec


@given(name=hst.text(min_size=1),
       sp1=hst.text(min_size=1), sp2=hst.text(min_size=1),
       inff1=hst.text(min_size=1), inff2=hst.text(min_size=1))
def test_creation(name, sp1, sp2, inff1, inff2):

    invalid_types = ['np.array', 'ndarray', 'lala', '', Number,
                     ndarray, 0, None]
    for inv_type in invalid_types:
        with pytest.raises(ValueError):
            ParamSpec(name, inv_type)

    ps = ParamSpec(name, 'real', label=None, unit='V',
                   inferred_from=(inff1, inff2),
                   depends_on=(sp1, sp2))
    assert ps.inferred_from == f'{inff1}, {inff2}'
    assert ps.depends_on == f'{sp1}, {sp2}'

    ps1 = ParamSpec(sp1, 'real')
    p1 = ParamSpec(name, 'real', depends_on=(ps1, sp2))
    assert p1.depends_on == ps.depends_on
