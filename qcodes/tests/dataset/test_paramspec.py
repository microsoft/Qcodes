import pytest
from numbers import Number

from numpy import ndarray
from hypothesis import given
import hypothesis.strategies as hst

from qcodes.dataset.param_spec import ParamSpec


@pytest.fixture
def version_0_serializations():
    sers = []
    sers.append({'name': 'dmm_v1',
                 'paramtype': 'numeric',
                 'label': 'Gate v1',
                 'unit': 'V',
                 'inferred_from': [],
                 'depends_on': ['dac_ch1', 'dac_ch2']})
    sers.append({'name': 'some_name',
                 'paramtype': 'array',
                 'label': 'My Array ParamSpec',
                 'unit': 'Ars',
                 'inferred_from': ['p1', 'p2' ],
                 'depends_on': []})
    return sers


@pytest.fixture
def version_0_deserializations():
    """
    The paramspecs that the above serializations should deserialize to
    """
    ps = []
    ps.append(ParamSpec('dmm_v1', paramtype='numeric', label='Gate v1',
                        unit='V', inferred_from=[],
                        depends_on=['dac_ch1', 'dac_ch2']))
    ps.append(ParamSpec('some_name', paramtype='array',
                        label='My Array ParamSpec', unit='Ars',
                        inferred_from=['p1', 'p2'], depends_on=[]))
    return ps


@given(name=hst.text(min_size=1),
       sp1=hst.text(min_size=1), sp2=hst.text(min_size=1),
       inff1=hst.text(min_size=1), inff2=hst.text(min_size=1),
       paramtype=hst.lists(
           elements=hst.sampled_from(['numeric', 'array', 'text']),
           min_size=6, max_size=6))
def test_creation(name, sp1, sp2, inff1, inff2, paramtype):
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
            ps = ParamSpec(name, paramtype[0], label=None, unit='V',
                           inferred_from=(inff1, inff2),
                           depends_on=(sp1, sp2))
        name = 'name'

    ps = ParamSpec(name, paramtype[1], label=None, unit='V',
                   inferred_from=(inff1, inff2),
                   depends_on=(sp1, sp2))

    assert ps.inferred_from == f'{inff1}, {inff2}'
    assert ps.depends_on == f'{sp1}, {sp2}'

    ps1 = ParamSpec(sp1, paramtype[2])
    p1 = ParamSpec(name, paramtype[3], depends_on=(ps1, sp2))
    assert p1.depends_on == ps.depends_on

    ps2 = ParamSpec(inff1, paramtype[4])
    p2 = ParamSpec(name, paramtype[5], inferred_from=(ps2, inff2))
    assert p2.inferred_from == ps.inferred_from


@given(name=hst.text(min_size=1))
def test_repr(name):
    okay_types = ['array', 'numeric', 'text']

    for okt in okay_types:
        if name.isidentifier():
            ps = ParamSpec(name, okt)
            expected_repr = (f"ParamSpec('{name}', '{okt}', '', '', "
                             "inferred_from=[], depends_on=[])")
            assert ps.__repr__() == expected_repr
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
    ps = ParamSpec(name3, "numeric", depends_on=[ps_indep])
    ps_copy = ps.copy()

    att_names = ["name", "type", "label", "unit",
                 "_inferred_from", "_depends_on"]

    attributes = {}
    for att in att_names:
        val = getattr(ps, att)
        valc = getattr(ps_copy, att)
        assert val == valc
        attributes[att] = val

    # Modifying the copy should not change the original
    for att in att_names:
        if not att.startswith('_'):
            setattr(ps_copy, att, attributes[att] + "_modified")
        else:
            setattr(ps_copy, att, attributes[att] + ['bob'])
        assert getattr(ps, att) == attributes[att]


def test_serialize():

    p1 = ParamSpec('p1', 'numeric', 'paramspec one', 'no unit',
                   depends_on=['some', 'thing'], inferred_from=['bab', 'bob'])

    ser = p1.serialize()

    assert ser['name'] == p1.name
    assert ser['paramtype'] == p1.type
    assert ser['label'] == p1.label
    assert ser['unit'] == p1.unit
    assert ser['depends_on'] == p1._depends_on
    assert ser['inferred_from'] == p1._inferred_from


def test_deserialize(version_0_serializations, version_0_deserializations):

    for sdict, ps in zip(version_0_serializations, version_0_deserializations):
        deps = ParamSpec.deserialize(sdict)
        assert ps == deps
