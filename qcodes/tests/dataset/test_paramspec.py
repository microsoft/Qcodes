from keyword import iskeyword
from numbers import Number

import pytest
from numpy import ndarray
from hypothesis import given, assume
import hypothesis.strategies as hst

from qcodes.dataset.param_spec import ParamSpec, ParamSpecBase


def valid_identifier(**kwargs):
    """Return a strategy which generates a valid Python Identifier"""
    if 'min_size' not in kwargs:
        kwargs['min_size'] = 4
    return hst.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
        **kwargs).filter(
        lambda x: x[0].isalpha() and x.isidentifier() and not (iskeyword(x))
    )


# This strategy generates a dict of kwargs needed to instantiate a valid
# ParamSpec object
valid_paramspec_kwargs = hst.fixed_dictionaries(
    {'name': valid_identifier(min_size=1, max_size=6),
     'paramtype': hst.sampled_from(['numeric', 'array', 'text']),
     'label': hst.one_of(hst.none(), hst.text(min_size=0, max_size=6)),
     'unit': hst.one_of(hst.none(), hst.text(min_size=0, max_size=2)),
     'depends_on': hst.lists(hst.text(min_size=1, max_size=3),
                             min_size=0, max_size=3),
     'inferred_from': hst.lists(hst.text(min_size=1, max_size=3),
                                min_size=0, max_size=3)
     })


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
                 'inferred_from': ['p1', 'p2'],
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
def test_depends_on(name1, name2, name3):
    ps2 = ParamSpec(name2, "numeric")
    ps3 = ParamSpec(name3, "numeric")

    ps1 = ParamSpec(name1, "numeric", depends_on=[ps2, ps3, 'foo'])

    assert ps1.depends_on == f"{ps2.name}, {ps3.name}, foo"
    assert ps1.depends_on_ == [ps2.name, ps3.name, "foo"]

    with pytest.raises(ValueError,
                       match=f"ParamSpec {name1} got string foo as depends_on. "
                       "It needs a Sequence of ParamSpecs or strings"):
        ParamSpec(name1, "numeric", depends_on='foo')


@given(
    name1=hst.text(min_size=4, alphabet=alphabet),
    name2=hst.text(min_size=4, alphabet=alphabet),
    name3=hst.text(min_size=4, alphabet=alphabet)
)
def test_inferred_from(name1, name2, name3):
    ps2 = ParamSpec(name2, "numeric")
    ps3 = ParamSpec(name3, "numeric")

    ps1 = ParamSpec(name1, "numeric", inferred_from=[ps2, ps3, 'bar'])

    assert ps1.inferred_from == f"{ps2.name}, {ps3.name}, bar"
    assert ps1.inferred_from_ == [ps2.name, ps3.name, "bar"]

    with pytest.raises(ValueError,
                       match=f"ParamSpec {name1} got string foo as "
                       f"inferred_from. "
                       "It needs a Sequence of ParamSpecs or strings"):
        ParamSpec(name1, "numeric", inferred_from='foo')


@given(
    name1=hst.text(min_size=4, alphabet=alphabet),
    name2=hst.text(min_size=4, alphabet=alphabet)
)
def test_copy(name1, name2):
    ps_indep = ParamSpec(name1, "numeric")
    ps = ParamSpec(name2, "numeric", depends_on=[ps_indep, 'other_param'])
    ps_copy = ps.copy()

    assert ps_copy == ps
    assert hash(ps_copy) == hash(ps)

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

    assert ps_copy != ps
    assert hash(ps_copy) != hash(ps)


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


@given(paramspecs=hst.lists(valid_paramspec_kwargs, min_size=2, max_size=2))
def test_hash(paramspecs):
    p1 = ParamSpec(**paramspecs[0])
    p2 = ParamSpec(**paramspecs[1])

    # call __hash__
    p1_h = hash(p1)
    p2_h = hash(p2)

    # make a set
    p_set = {p1, p2}

    # test that the hash equality follows object equality
    if p1 == p2:
        assert p1_h == p2_h
        assert 1 == len(p_set)
    else:
        assert p1_h != p2_h
        assert 2 == len(p_set)


@given(paramspecs=hst.lists(valid_paramspec_kwargs, min_size=6, max_size=6),
       add_to_1_inf=hst.booleans(),
       add_to_1_dep=hst.booleans(),
       add_to_2_inf=hst.booleans(),
       add_to_2_dep=hst.booleans(),
       )
def test_hash_with_deferred_and_inferred_as_paramspecs(
        paramspecs, add_to_1_inf, add_to_1_dep, add_to_2_inf, add_to_2_dep):
    """
    Test that hashing works if 'inferred_from' and/or 'depends_on' contain
    actual ParamSpec instances and not just strings.
    """
    assume(add_to_1_inf or add_to_1_dep or add_to_2_inf or add_to_2_dep)

    # Add ParamSpecs to 'inferred_from' and/or 'depend_on' lists next to
    # strings (that are generated by the main strategy)
    if add_to_1_inf:
        paramspecs[0]['inferred_from'].append(ParamSpec(**paramspecs[2]))
    if add_to_1_dep:
        paramspecs[0]['depends_on'].append(ParamSpec(**paramspecs[3]))
    if add_to_2_inf:
        paramspecs[1]['inferred_from'].append(ParamSpec(**paramspecs[4]))
    if add_to_2_dep:
        paramspecs[1]['depends_on'].append(ParamSpec(**paramspecs[5]))

    p1 = ParamSpec(**paramspecs[0])
    p2 = ParamSpec(**paramspecs[1])

    # call __hash__
    p1_h = hash(p1)
    p2_h = hash(p2)

    # make a set
    p_set = {p1, p2}

    # test that the hash equality follows object equality
    if p1 == p2:
        assert p1_h == p2_h
        assert 1 == len(p_set)
    else:
        assert p1_h != p2_h
        assert 2 == len(p_set)


@given(paramspecs=hst.lists(valid_paramspec_kwargs, min_size=1, max_size=1))
def test_base_version(paramspecs):

    kwargs = paramspecs[0]

    ps = ParamSpec(**kwargs)
    ps_base = ParamSpecBase(name=kwargs['name'],
                            paramtype=kwargs['paramtype'],
                            label=kwargs['label'],
                            unit=kwargs['unit'])

    assert ps.base_version() == ps_base
