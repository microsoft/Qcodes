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

    if not inff1.isidentifier():
        inff1 = 'inff1'

    if not sp1.isidentifier():
        sp1 = 'sp1'

    if not name.isidentifier():
        with pytest.raises(ValueError):
            ps = ParamSpec(name, 'numeric', label=None, unit='V',
                           inferred_from=(inff1, inff2),
                           depends_on=(sp1, sp2))
        name = 'name'

    ps = ParamSpec(name, 'numeric', label=None, unit='V',
                   inferred_from=(inff1, inff2),
                   depends_on=(sp1, sp2))

    assert ps.inferred_from == f'{inff1}, {inff2}'
    assert ps.depends_on == f'{sp1}, {sp2}'

    ps1 = ParamSpec(sp1, 'numeric')
    p1 = ParamSpec(name, 'numeric', depends_on=(ps1, sp2))
    assert p1.depends_on == ps.depends_on

    ps2 = ParamSpec(inff1, 'numeric')
    p2 = ParamSpec(name, 'numeric', inferred_from=(ps2, inff2))
    assert p2.inferred_from == ps.inferred_from


@given(name=hst.text(min_size=1))
def test_repr(name):
    okay_types = ['array', 'numeric', 'text']

    for okt in okay_types:
        if name.isidentifier():
            ps = ParamSpec(name, okt)
            assert ps.__repr__() == f"{name} ({okt})"
        else:
            with pytest.raises(ValueError):
                ps = ParamSpec(name, okt)

alphabet = "".join([chr(i) for i in range(ord("a"), ord("z"))])

@given(
    name1=hst.text(min_size=4, alphabet=alphabet),
    name2=hst.text(min_size=4, alphabet=alphabet),
    name3=hst.text(min_size=4, alphabet=alphabet)
)
def test_add_depends_on(name1, name2, name3):

    ps1 = ParamSpec(name1, "numeric")
    ps2 = ParamSpec(name2, "numeric")
    ps3 = ParamSpec(name3, "numeric")
    ps1.add_depends_on([ps2, ps3])

    assert ps1.depends_on == f"{ps2.name}, {ps3.name}"


@given(
    name1=hst.text(min_size=4, alphabet=alphabet),
    name2=hst.text(min_size=4, alphabet=alphabet),
    name3=hst.text(min_size=4, alphabet=alphabet)
)
def test_add_inferred_from(name1, name2, name3):

    ps1 = ParamSpec(name1, "numeric")
    ps2 = ParamSpec(name2, "numeric")
    ps3 = ParamSpec(name3, "numeric")
    ps1.add_inferred_from([ps2, ps3])

    assert ps1.inferred_from == f"{ps2.name}, {ps3.name}"


@given(
    name1=hst.text(min_size=4, alphabet=alphabet),
    name2=hst.text(min_size=4, alphabet=alphabet),
    name3=hst.text(min_size=4, alphabet=alphabet),
)
def test_copy(name1, name2, name3):

    ps_indep = ParamSpec(name1, "numeric")
    ps_indep_2 = ParamSpec(name2, "numeric")
    ps = ParamSpec(name3, "numeric", depends_on=[ps_indep])
    ps_copy = ps.copy()

    attributes = {}
    for att in ["name", "type", "label", "unit"]:
        val = getattr(ps, att)
        valc = getattr(ps_copy, att)
        assert val == valc
        attributes[att] = val

    # Modifying the copy should not change the original
    for att in ["name", "type", "label", "unit"]:
        setattr(ps_copy, att, attributes[att] + "_modified")
        assert getattr(ps, att) == attributes[att]

    ps_copy.add_depends_on([ps_indep_2])
    assert ps_copy.depends_on == f"{ps_indep.name}, {ps_indep_2.name}"
    assert ps.depends_on == f"{ps_indep.name}"
