import json
import re
from copy import deepcopy

import pytest

from qcodes.dataset.descriptions.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.descriptions.dependencies import (DependencyError,
                                                      InferenceError,
                                                      InterDependencies_)
from qcodes.dataset.descriptions.versioning.v0 import InterDependencies
from qcodes.dataset.descriptions.versioning.converters import (                     new_to_old, old_to_new)
from qcodes.tests.common import error_caused_by
# pylint: disable=unused-import
from qcodes.tests.dataset.interdeps_fixtures import (some_interdeps,
                                                     some_paramspecs,
                                                     some_paramspecbases)



def test_wrong_input_raises():

    for pspecs in [['p1', 'p2', 'p3'],
                   [ParamSpec('p1', paramtype='numeric'), 'p2'],
                   ['p1', ParamSpec('p2', paramtype='text')]]:

        with pytest.raises(ValueError):
            InterDependencies(pspecs)


def test_init(some_paramspecbases):
    """
    Assert that the init functions correctly sets up the object.
    Assert via the public-facing methods.
    """

    (ps1, ps2, ps3, ps4) = some_paramspecbases

    idps1 = InterDependencies_(dependencies={ps1: (ps2,)})
    idps2 = InterDependencies_(dependencies={ps1: (ps2, ps2, ps2)})

    assert idps1 == idps2
    assert idps1.what_depends_on(ps2) == (ps1,)
    assert idps1.what_is_inferred_from(ps2) == ()
    assert idps1.non_dependencies == (ps1,)

    idps1 = InterDependencies_(dependencies={ps1: (ps2, ps3)})
    idps2 = InterDependencies_(dependencies={ps1: (ps3, ps2)})

    assert idps1.what_depends_on(ps2) == (ps1,)
    assert idps1.what_depends_on(ps3) == (ps1,)
    assert idps1.non_dependencies == (ps1,)
    assert idps2.non_dependencies == (ps1,)

    idps = InterDependencies_(dependencies={ps1: (ps3, ps2),
                                            ps4: (ps3,)})
    assert set(idps.what_depends_on(ps3)) == set((ps1, ps4))
    assert idps.non_dependencies == (ps1, ps4)


def test_init_validation_raises(some_paramspecbases):

    (ps1, ps2, ps3, ps4) = some_paramspecbases

    # First test validation of trees invalid in their own right

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

    # Now test trees that are invalid together

    invalid_trees = [{'deps': {ps1: (ps2, ps3)},
                      'inffs': {ps2: (ps4, ps1)}}]
    for inv in invalid_trees:
        with pytest.raises(ValueError,
                           match=re.escape("Invalid dependencies/inferences")):
            InterDependencies_(dependencies=inv['deps'],
                               inferences=inv['inffs'])

def test_to_dict(some_paramspecbases):

    def tester(idps):
        ser = idps._to_dict()
        json.dumps(ser)
        idps_deser = InterDependencies_._from_dict(ser)
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


def test_old_to_new_and_back(some_paramspecs):

    idps_old = InterDependencies(*some_paramspecs[1].values())
    idps_new = old_to_new(idps_old)

    assert new_to_old(idps_new) == idps_old


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


    assert idps_new.dependencies == {}
    assert idps_new.inferences == {ps3_base: (ps1_base,)}
    assert idps_new.standalones == set((ps2_base,))
    paramspecs = (ps1_base, ps2_base, ps3_base)
    assert idps_new._id_to_paramspec == {ps.name: ps for ps in paramspecs}

    idps_old = InterDependencies(ps2, ps4, ps1, ps2, ps3, ps5, ps6)

    idps_new = old_to_new(idps_old)

    assert idps_new.dependencies == {ps5_base: (ps3_base, ps4_base),
                                     ps6_base: (ps3_base, ps4_base)}
    assert idps_new.inferences == {ps3_base: (ps1_base,),
                                   ps4_base: (ps2_base,)}
    assert idps_new.standalones == set()
    paramspecs = (ps1_base, ps2_base, ps3_base, ps4_base, ps5_base, ps6_base)
    assert idps_new._id_to_paramspec == {ps.name: ps for ps in paramspecs}

    idps_old = InterDependencies(ps1, ps2)

    idps_new = old_to_new(idps_old)

    assert idps_new.dependencies == {}
    assert idps_new.inferences == {}
    assert idps_new.standalones == set((ps1_base, ps2_base))
    paramspecs = (ps1_base, ps2_base)
    assert idps_new._id_to_paramspec == {ps.name: ps for ps in paramspecs}


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
    idps_old_expected = InterDependencies(paramspec2, paramspec3,
                                          paramspec1, paramspec4)

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
    idps_old_expected = InterDependencies(paramspec2, paramspec3,
                                          paramspec1, paramspec4)

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

    with pytest.raises(DependencyError) as exc_info:
        idps.validate_subset((ps1,))
    assert exc_info.value._param_name == 'psb1'
    assert exc_info.value._missing_params == {'psb2', 'psb3'}

    with pytest.raises(DependencyError) as exc_info:
        idps.validate_subset((ps1, ps2, ps4))
    assert exc_info.value._param_name == 'psb1'
    assert exc_info.value._missing_params == {'psb3'}

    with pytest.raises(InferenceError) as exc_info:
        idps.validate_subset((ps3,))
    assert exc_info.value._param_name == 'psb3'
    assert exc_info.value._missing_params == {'psb4'}

    with pytest.raises(InferenceError) as exc_info:
        idps2 = InterDependencies_(dependencies={ps1: (ps2, ps3)},
                                    inferences={ps3: (ps4,)})
        idps2.validate_subset((ps1, ps2, ps3))
    assert exc_info.value._param_name == 'psb3'
    assert exc_info.value._missing_params == {'psb4'}

    with pytest.raises(ValueError, match='ps42'):
        ps42 = ParamSpecBase('ps42', paramtype='text', label='', unit='it')
        idps.validate_subset((ps2, ps42, ps4))


def test_extend(some_paramspecbases):

    ps1, ps2, ps3, _ = some_paramspecbases

    idps = InterDependencies_(standalones=(ps1, ps2))

    idps_ext = idps.extend(dependencies={ps1: (ps3,)})
    idps_expected = InterDependencies_(standalones=(ps2,),
                                       dependencies={ps1: (ps3,)})
    assert idps_ext == idps_expected

    # lazily check that we get brand new objects
    idps._id_to_paramspec[ps1.name].label = "Something new and awful"
    idps._id_to_paramspec[ps2.name].unit = "Ghastly unit"
    assert idps_ext._id_to_paramspec[ps1.name].label == 'blah'
    assert idps_ext._id_to_paramspec[ps2.name].unit == 'V'
    # reset the objects that are never supposed to be mutated
    idps._id_to_paramspec[ps1.name].label = "blah"
    idps._id_to_paramspec[ps2.name].unit = "V"

    idps = InterDependencies_(standalones=(ps2,))
    idps_ext = idps.extend(dependencies={ps1: (ps2,)})
    idps_expected = InterDependencies_(dependencies={ps1: (ps2,)})
    assert idps_ext == idps_expected

    idps = InterDependencies_(dependencies={ps1: (ps2,)})
    idps_ext = idps.extend(dependencies={ps1: (ps2, ps3)})
    idps_expected = InterDependencies_(dependencies={ps1: (ps2, ps3)})
    assert idps_ext == idps_expected

    idps = InterDependencies_()
    idps_ext = idps.extend(standalones=(ps1, ps2))
    idps_expected = InterDependencies_(standalones=(ps2, ps1))
    assert idps_ext == idps_expected

    ps_nu = deepcopy(ps1)
    ps_nu.unit += '/s'
    idps = InterDependencies_(standalones=(ps1,))
    idps_ext = idps.extend(standalones=(ps_nu,))
    idps_expected = InterDependencies_(standalones=(ps_nu, ps1))
    assert idps_ext == idps_expected

    idps = InterDependencies_(dependencies={ps1: (ps2,)})
    match = re.escape("Invalid dependencies/inferences")
    with pytest.raises(ValueError, match=match):
        idps_ext = idps.extend(inferences={ps2: (ps1,)})


def test_remove(some_paramspecbases):
    ps1, ps2, ps3, ps4 = some_paramspecbases

    idps = InterDependencies_(dependencies={ps1: (ps2, ps3)},
                              inferences={ps2: (ps4, )})
    idps_rem = idps.remove(ps1)
    idps_expected = InterDependencies_(inferences={ps2: (ps4,)},
                                       standalones=(ps3,))
    assert idps_rem == idps_expected

    for p in [ps4, ps2, ps3]:
        match = re.escape(f'Cannot remove {p.name}, other parameters')
        with pytest.raises(ValueError, match=match):
            idps_rem = idps.remove(p)

    idps = InterDependencies_(dependencies={ps1: (ps3,)},
                              inferences={ps2: (ps4,)})
    idps_rem = idps.remove(ps2)
    idps_expected = InterDependencies_(dependencies={ps1: (ps3,)},
                                       standalones=(ps4,))

    assert idps_rem == idps_expected

    idps = InterDependencies_(dependencies={ps1: (ps2, ps3)},
                              standalones=(ps4, ))
    idps_rem = idps.remove(ps4)
    idps_expected = InterDependencies_(dependencies={ps1: (ps2, ps3)})
    assert idps_rem == idps_expected

    idps = InterDependencies_(dependencies={ps1: (ps2, ps3)},
                              standalones=(ps4, ))
    idps_rem = idps.remove(ps1)
    idps_expected = InterDependencies_(standalones=(ps2, ps3, ps4))
    assert idps_rem == idps_expected

def test_equality_old(some_paramspecs):

    # TODO: make this more fancy with itertools

    ps1 = some_paramspecs[1]['ps1']
    ps2 = some_paramspecs[1]['ps2']
    ps3 = some_paramspecs[1]['ps3']
    ps4 = some_paramspecs[1]['ps4']
    ps5 = some_paramspecs[1]['ps5']
    ps6 = some_paramspecs[1]['ps6']

    assert InterDependencies(ps1, ps2, ps3) == InterDependencies(ps3, ps2, ps1)
    assert InterDependencies(ps1, ps6, ps3) == InterDependencies(ps3, ps6, ps1)
    assert InterDependencies(ps4, ps5, ps3) == InterDependencies(ps3, ps4, ps5)


def test_non_dependents():
    ps1 = ParamSpecBase('ps1', paramtype='numeric', label='Raw Data 1',
                        unit='V')
    ps2 = ParamSpecBase('ps2', paramtype='array', label='Raw Data 2',
                        unit='V')
    ps3 = ParamSpecBase('ps3', paramtype='text', label='Axis 1',
                        unit='')
    ps4 = ParamSpecBase('ps4', paramtype='numeric', label='Axis 2',
                        unit='V')
    ps5 = ParamSpecBase('ps5', paramtype='numeric', label='Signal',
                        unit='Conductance')
    ps6 = ParamSpecBase('ps6', paramtype='text', label='Goodness',
                        unit='')

    idps1 = InterDependencies_(dependencies={ps5: (ps3, ps4), ps6: (ps3, ps4)},
                               inferences={ps4: (ps2,), ps3: (ps1,)})

    assert idps1.non_dependencies == (ps5, ps6)

    idps2 = InterDependencies_(dependencies={ps2: (ps1,)})

    assert idps2.non_dependencies == (ps2,)

    idps3 = InterDependencies_(dependencies={ps6: (ps1,)},
                               standalones=(ps2,))

    assert idps3.non_dependencies == (ps2, ps6)
