import json

import pytest

from qcodes.dataset.dependencies import (InterDependencies,
                                         InterDependencies_,
                                         old_to_new,
                                         new_to_old,
                                         DependencyError,
                                         InferenceError)
from qcodes.dataset.param_spec import ParamSpec, ParamSpecBase
from qcodes.tests.common import error_caused_by
# pylint: disable=unused-import
from qcodes.tests.dataset.test_descriptions import some_paramspecs

@pytest.fixture
def some_paramspecbases():

    psb1 = ParamSpecBase('psb1', paramtype='text', label='blah', unit='')
    psb2 = ParamSpecBase('psb2', paramtype='array', label='', unit='V')
    psb3 = ParamSpecBase('psb3', paramtype='array', label='', unit='V')
    psb4 = ParamSpecBase('psb4', paramtype='numeric', label='number', unit='')

    return (psb1, psb2, psb3, psb4)


def test_wrong_input_raises():

    for pspecs in [['p1', 'p2', 'p3'],
                   [ParamSpec('p1', paramtype='numeric'), 'p2'],
                   ['p1', ParamSpec('p2', paramtype='text')]]:

        with pytest.raises(ValueError):
            InterDependencies(pspecs)


def test_init_validation_raises(some_paramspecbases):

    (ps1, ps2, _, _) = some_paramspecbases

    invalid_trees = ([ps1, ps2],
                     {'ps1': 'ps2'},
                     {ps1: 'ps2'},
                     {ps1: ('ps2',)},
                     {ps1: (ps2,), ps2: (ps1,)}
                     )
    causes = ("ParamSpecTree must be a dict",
              "ParamSpecTree must have ParamSpecs as keys",
              "ParamSpecTree must have tuple values",
              "ParamSpecTree can only have tuples "
              "of ParamSpecs as values",
              "ParamSpecTree can not have cycles")

    for tree, cause in zip(invalid_trees, causes):
        with pytest.raises(ValueError, match='Invalid dependencies') as ei:
            InterDependencies_(dependencies=tree, inferences={})

        assert error_caused_by(ei, cause=cause)

    for tree, cause in zip(invalid_trees, causes):
        with pytest.raises(ValueError, match='Invalid inferences') as ei:
            InterDependencies_(dependencies={}, inferences=tree)

        assert error_caused_by(ei, cause=cause)

    with pytest.raises(ValueError, match='Invalid standalones') as ei:
        InterDependencies_(standalones=('ps1', 'ps2'))

    assert error_caused_by(ei, cause='Standalones must be a sequence of '
                                     'ParamSpecs')


def test_serialize(some_paramspecbases):

    def tester(idps):
        ser = idps.serialize()
        json.dumps(ser)
        idps_deser = InterDependencies_.deserialize(ser)
        assert idps == idps_deser

    (ps1, ps2, ps3, ps4) = some_paramspecbases

    idps = InterDependencies_(standalones=(ps1, ps2),
                              dependencies={ps3: (ps4,)})
    tester(idps)

    idps = InterDependencies_(standalones=(ps1, ps2, ps3, ps4))
    tester(idps)

    idps = InterDependencies_(dependencies={ps1: (ps2, ps3)},
                              inferences={ps2: (ps4,), ps3: (ps4,)})
    tester(idps)


def test_old_to_new(some_paramspecs):

    ps1 = some_paramspecs[1]['ps1']
    ps2 = some_paramspecs[1]['ps2']
    ps3 = some_paramspecs[1]['ps3']
    ps4 = some_paramspecs[1]['ps4']
    ps5 = some_paramspecs[1]['ps5']
    ps6 = some_paramspecs[1]['ps6']

    idps_old = InterDependencies(ps1, ps2, ps3)

    idps_new = old_to_new(idps_old)

    ps1_base = ps1.base_version()
    ps2_base = ps2.base_version()
    ps3_base = ps3.base_version()
    ps4_base = ps4.base_version()
    ps5_base = ps5.base_version()
    ps6_base = ps6.base_version()

    _id = idps_new._id

    assert idps_new.dependencies == {}
    assert idps_new.inferences == {ps3_base: (ps1_base,)}
    assert idps_new.standalones == set((ps2_base,))
    paramspecs = (ps1_base, ps2_base, ps3_base)
    assert idps_new._id_to_paramspec == {_id(ps): ps for ps in paramspecs}

    idps_old = InterDependencies(ps2, ps4, ps1, ps2, ps3, ps5, ps6)

    idps_new = old_to_new(idps_old)

    assert idps_new.dependencies == {ps5_base: (ps3_base, ps4_base),
                                     ps6_base: (ps3_base, ps4_base)}
    assert idps_new.inferences == {ps3_base: (ps1_base,),
                                   ps4_base: (ps2_base,)}
    assert idps_new.standalones == set()
    paramspecs = (ps1_base, ps2_base, ps3_base, ps4_base, ps5_base, ps6_base)
    assert idps_new._id_to_paramspec == {_id(ps): ps for ps in paramspecs}

    idps_old = InterDependencies(ps1, ps2)

    idps_new = old_to_new(idps_old)

    assert idps_new.dependencies == {}
    assert idps_new.inferences == {}
    assert idps_new.standalones == set((ps1_base, ps2_base))
    paramspecs = (ps1_base, ps2_base)
    assert idps_new._id_to_paramspec == {_id(ps): ps for ps in paramspecs}


def test_new_to_old(some_paramspecbases):

    (ps1, ps2, ps3, ps4) = some_paramspecbases

    idps_new = InterDependencies_(dependencies={ps1: (ps2, ps3)},
                                  standalones=(ps4,))

    paramspec1 = ParamSpec(name=ps1.name, paramtype=ps1.type,
                           label=ps1.label, unit=ps1.unit,
                           depends_on=[ps2.name, ps3.name])
    paramspec2 = ParamSpec(name=ps2.name, paramtype=ps2.type,
                           label=ps2.label, unit=ps2.unit)
    paramspec3 = ParamSpec(name=ps3.name, paramtype=ps3.type,
                           label=ps3.label, unit=ps3.unit)
    paramspec4 = ParamSpec(name=ps4.name, paramtype=ps4.type,
                           label=ps4.label, unit=ps4.unit)
    idps_old_expected = InterDependencies(paramspec1, paramspec2,
                                          paramspec3, paramspec4)

    assert new_to_old(idps_new) == idps_old_expected

    #

    idps_new = InterDependencies_(inferences={ps1: (ps2, ps3)},
                                  standalones=(ps4,))

    paramspec1 = ParamSpec(name=ps1.name, paramtype=ps1.type,
                           label=ps1.label, unit=ps1.unit,
                           inferred_from=[ps2.name, ps3.name])
    paramspec2 = ParamSpec(name=ps2.name, paramtype=ps2.type,
                           label=ps2.label, unit=ps2.unit)
    paramspec3 = ParamSpec(name=ps3.name, paramtype=ps3.type,
                           label=ps3.label, unit=ps3.unit)
    paramspec4 = ParamSpec(name=ps4.name, paramtype=ps4.type,
                           label=ps4.label, unit=ps4.unit)
    idps_old_expected = InterDependencies(paramspec1, paramspec2,
                                          paramspec3, paramspec4)

    assert new_to_old(idps_new) == idps_old_expected


def test_extend_with_paramspec(some_paramspecs):
    ps1 = some_paramspecs[1]['ps1']
    ps2 = some_paramspecs[1]['ps2']
    ps3 = some_paramspecs[1]['ps3']
    ps4 = some_paramspecs[1]['ps4']
    ps5 = some_paramspecs[1]['ps5']
    ps6 = some_paramspecs[1]['ps6']

    ps1_base = ps1.base_version()
    ps2_base = ps2.base_version()
    ps3_base = ps3.base_version()
    ps4_base = ps4.base_version()
    ps5_base = ps5.base_version()
    ps6_base = ps6.base_version()

    idps_bare = InterDependencies_(standalones=(ps1_base,))
    idps_extended = InterDependencies_(inferences={ps3_base: (ps1_base,)})

    assert idps_bare._extend_with_paramspec(ps3) == idps_extended

    idps_bare = InterDependencies_(standalones=(ps2_base,),
                                   inferences={ps3_base: (ps1_base,)})
    idps_extended = InterDependencies_(inferences={ps3_base: (ps1_base,),
                                                   ps4_base: (ps2_base,)})

    assert idps_bare._extend_with_paramspec(ps4) == idps_extended

    idps_bare = InterDependencies_(standalones=(ps1_base, ps2_base))
    idps_extended = InterDependencies_(
                        inferences={ps3_base: (ps1_base,),
                                    ps4_base: (ps2_base,)},
                        dependencies={ps5_base: (ps3_base, ps4_base),
                                      ps6_base: (ps3_base, ps4_base)})
    assert (idps_bare.
            _extend_with_paramspec(ps3).
            _extend_with_paramspec(ps4).
            _extend_with_paramspec(ps5).
            _extend_with_paramspec(ps6)) == idps_extended


def test_validate_subset(some_paramspecbases):

    ps1, ps2, ps3, ps4 = some_paramspecbases

    idps = InterDependencies_(dependencies={ps1: (ps2, ps3)},
                              inferences={ps2: (ps4,), ps3: (ps4,)})

    idps.validate_subset((ps4,))
    idps.validate_subset((ps2, ps4))
    idps.validate_subset((ps2, ps3, ps4))
    idps.validate_subset(())
    idps.validate_subset([])

    with pytest.raises(DependencyError):
        idps.validate_subset((ps1,))

    with pytest.raises(InferenceError):
        idps.validate_subset((ps2, ps3))

    with pytest.raises(InferenceError):
        idps.validate_subset((ps1, ps2, ps3))