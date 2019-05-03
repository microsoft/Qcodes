import json
import re
from copy import deepcopy
from itertools import permutations, chain

import pytest

from qcodes.dataset.dependencies import (InterDependencies,
                                         InterDependencies_,
                                         ParamSpecTree,
                                         ParamSpecGrove,
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

@pytest.fixture
def some_interdeps():
    """
    Some different InterDependencies_ objects for testing
    """
    idps_list = []
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

    idps = InterDependencies_(dependencies={ps5: (ps3, ps4), ps6: (ps3, ps4)},
                              inferences={ps4: (ps2,), ps3: (ps1,)})

    idps_list.append(idps)

    ps1 = ParamSpecBase('ps1', paramtype='numeric',
                        label='setpoint', unit='Hz')
    ps2 = ParamSpecBase('ps2', paramtype='numeric', label='signal',
                        unit='V')
    idps = InterDependencies_(dependencies={ps2: (ps1,)})

    idps_list.append(idps)

    return idps_list


def test_tree_and_grove_iteration(some_paramspecbases):

    Tree = ParamSpecTree

    (ps1, ps2, ps3, ps4) = some_paramspecbases

    psbs = [(ps1,), (ps2, ps3), (ps4,)]

    trees = tuple(Tree(*psb) for psb in psbs)

    grove = ParamSpecGrove(*trees)

    for actual_tree, expected_tree, psb_tup in zip(grove, trees, psbs):
        assert actual_tree == expected_tree
        for actual_psb, expected_psb in zip(actual_tree, psb_tup):
            assert actual_psb == expected_psb


def test_tre_serialization(some_paramspecbases):

    Tree = ParamSpecTree

    (ps1, ps2, ps3, ps4) = some_paramspecbases


    trees = (Tree(ps1), Tree(ps2, ps1), Tree(ps1, ps2, ps3, ps4),
             Tree(ps1, ps2))

    for tree in trees:
        assert Tree.deserialize(tree.serialize()) == tree


def test_grove_serialization(some_paramspecbases):

    Tree = ParamSpecTree

    (ps1, ps2, ps3, ps4) = some_paramspecbases

    trees = (Tree(ps1), Tree(ps2, ps3), Tree(ps4))

    all_tree_combos = chain(*map(lambda n: permutations(trees, n),
                                 range(1, len(trees) + 1)))

    for tree_combo in all_tree_combos:
        grove = ParamSpecGrove(*tree_combo)
        assert ParamSpecGrove.deserialize(grove.serialize()) == grove


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

    Tree = ParamSpecTree

    (ps1, ps2, ps3, ps4) = some_paramspecbases

    idps1 = InterDependencies_(dependencies=(Tree(ps1, ps2),))
    idps2 = InterDependencies_(inferences=(Tree(ps1, ps2),))

    assert idps1.dependencies == idps2.inferences
    assert idps1.what_depends_on(ps2) == (ps1,)
    assert idps1.what_is_inferred_from(ps2) == ()
    assert idps2.what_depends_on(ps2) == ()
    assert idps2.what_is_inferred_from(ps2) == (ps1,)

    idps1 = InterDependencies_(dependencies=(Tree(ps1, ps2, ps3),))
    idps2 = InterDependencies_(dependencies=(Tree(ps1, ps3, ps2),))

    assert idps1.what_depends_on(ps2) == (ps1,)
    assert idps1.what_depends_on(ps3) == (ps1,)

    idps = InterDependencies_(dependencies=(Tree(ps1, ps3, ps2),
                                            Tree(ps4, ps3)))
    assert set(idps.what_depends_on(ps3)) == set((ps1, ps4))


def test_init_validation_raises(some_paramspecbases):

    Tree = ParamSpecTree

    (ps1, ps2, ps3, ps4) = some_paramspecbases

    not_trees = [ps1, 'tree', 0, None, ()]

    for not_a_tree in not_trees:

        match = ('ParamSpecGrove can only contain '
                'ParamSpecTrees, but received a/an '
                f'{type(not_a_tree)} instead.')
        with pytest.raises(ValueError, match=match):
            InterDependencies_(dependencies=(ParamSpecTree(ps1, ps2),
                                             not_a_tree))

    with_cycles = [(Tree(ps1, ps2), Tree(ps2, ps1)),
                   (Tree(ps4, ps2, ps3), Tree(ps1, ps3), Tree(ps3))]

    for tup_wc in with_cycles:
        with pytest.raises(ValueError, match='Cycles detected'):
            InterDependencies_(dependencies=tup_wc)

    ps5 = ParamSpecBase(ps1.name, 'text', 'label', 'unit')

    match = "Supplied trees do not have unique root names"
    with pytest.raises(ValueError, match=match):
        InterDependencies_(inferences=(Tree(ps1, ps2, ps3),
                                       Tree(ps5, ps2, ps3)))

def test_serialize(some_paramspecbases):

    Tree = ParamSpecTree

    def tester(idps):
        ser = idps.serialize()
        json.dumps(ser)
        idps_deser = InterDependencies_.deserialize(ser)
        assert idps == idps_deser

    (ps1, ps2, ps3, ps4) = some_paramspecbases

    idps = InterDependencies_((Tree(ps3, ps4), Tree(ps1), Tree(ps2)))
    tester(idps)

    idps = InterDependencies_((Tree(ps1), Tree(ps2), Tree(ps3), Tree(ps4)))
    tester(idps)

    idps = InterDependencies_(dependencies=(Tree(ps1, ps2, ps3),),
                              inferences=(Tree(ps2, ps4), Tree(ps3, ps4)))
    tester(idps)


def test_old_to_new(some_paramspecs):

    Tree = ParamSpecTree

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


    assert idps_new.dependencies == ParamSpecGrove(Tree(ps2_base))
    assert idps_new.inferences == ParamSpecGrove(Tree(ps3_base, ps1_base))

    paramspecs = (ps1_base, ps2_base, ps3_base)
    assert idps_new._id_to_paramspec == {ps.name: ps for ps in paramspecs}

    idps_old = InterDependencies(ps2, ps4, ps1, ps2, ps3, ps5, ps6)

    idps_new = old_to_new(idps_old)

    assert idps_new.dependencies == ParamSpecGrove(
        Tree(ps5_base, ps3_base, ps4_base), Tree(ps6_base, ps3_base, ps4_base))
    assert idps_new.inferences == ParamSpecGrove(
        Tree(ps4_base, ps2_base), Tree(ps3_base, ps1_base))

    paramspecs = (ps1_base, ps2_base, ps3_base, ps4_base, ps5_base, ps6_base)
    assert idps_new._id_to_paramspec == {ps.name: ps for ps in paramspecs}

    idps_old = InterDependencies(ps1, ps2)

    idps_new = old_to_new(idps_old)

    assert idps_new.dependencies == ParamSpecGrove(Tree(ps1_base),
                                                   Tree(ps2_base))
    assert idps_new.inferences == ParamSpecGrove()
    paramspecs = (ps1_base, ps2_base)
    assert idps_new._id_to_paramspec == {ps.name: ps for ps in paramspecs}


def test_new_to_old(some_paramspecbases):

    Tree = ParamSpecTree

    (ps1, ps2, ps3, ps4) = some_paramspecbases

    idps_new = InterDependencies_(dependencies=(Tree(ps1, ps2, ps3),
                                                Tree(ps4,)))

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

    idps_new = InterDependencies_(inferences=(Tree(ps1, ps2, ps3),),
                                  dependencies=(Tree(ps4),))

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


def test_old_to_new_and_back(some_paramspecs):

    idps_old = InterDependencies(*some_paramspecs[1].values())
    idps_new = old_to_new(idps_old)

    assert new_to_old(idps_new) == idps_old


def test_validate_subset(some_paramspecbases):

    Tree = ParamSpecTree

    ps1, ps2, ps3, ps4 = some_paramspecbases

    idps = InterDependencies_(dependencies=(Tree(ps1, ps2, ps3),),
                              inferences=(Tree(ps2, ps4), Tree(ps3, ps4)))

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
        idps2 = InterDependencies_(dependencies=(Tree(ps1, ps2, ps3),),
                                    inferences=(Tree(ps3, ps4),))
        idps2.validate_subset((ps1, ps2, ps3))
    assert exc_info.value._param_name == 'psb3'
    assert exc_info.value._missing_params == {'psb4'}

    with pytest.raises(ValueError, match='ps42'):
        ps42 = ParamSpecBase('ps42', paramtype='text', label='', unit='it')
        idps.validate_subset((ps2, ps42, ps4))


def test_extend_with_tree(some_paramspecbases):

    Tree = ParamSpecTree

    ps1, ps2, ps3, _ = some_paramspecbases

    idps = InterDependencies_(dependencies=(Tree(ps1), Tree(ps2)))

    idps_ext = idps.extend_with_tree(Tree(ps1, ps3), 'deps')
    idps_expected = InterDependencies_(dependencies=(Tree(ps2),
                                                     Tree(ps1, ps3)))
    assert idps_ext == idps_expected

    idps = InterDependencies_(dependencies=(Tree(ps2),))
    idps_ext = idps.extend_with_tree(Tree(ps1, ps2), 'deps')
    idps_expected = InterDependencies_(dependencies=(Tree(ps1, ps2),))
    assert idps_ext == idps_expected

    # note: the following did not raise before we introduced trees

    idps = InterDependencies_(dependencies=(Tree(ps1, ps2),))
    match = 'Supplied trees do not have unique root names'
    with pytest.raises(ValueError, match=match):
        idps_ext = idps.extend_with_tree(Tree(ps1, ps2, ps3), 'deps')

    idps = InterDependencies_()
    idps_ext = idps.extend_with_tree(Tree(ps1), 'deps')
    idps_ext = idps_ext.extend_with_tree(Tree(ps2), 'deps')
    idps_expected = InterDependencies_(dependencies=(Tree(ps1), Tree(ps2)))
    assert idps_ext == idps_expected

    # note: the following did not raise before we introduced trees

    ps_nu = deepcopy(ps1)
    ps_nu.unit += '/s'
    idps = InterDependencies_(dependencies=(Tree(ps1),))
    match = 'Supplied trees do not have unique root names'
    with pytest.raises(ValueError, match=match):
        idps_ext = idps.extend_with_tree(Tree(ps_nu), 'deps')


    idps = InterDependencies_(dependencies=(Tree(ps1, ps2),))
    match = re.escape("Invalid dependencies/inferences")
    with pytest.raises(ValueError, match=match):
        idps_ext = idps.extend_with_tree(Tree(ps2, ps1), 'inffs')


def test_remove(some_paramspecbases):

    Tree = ParamSpecTree

    ps1, ps2, ps3, ps4 = some_paramspecbases

    idps = InterDependencies_(dependencies=(Tree(ps1, ps2, ps3),),
                              inferences=(Tree(ps2, ps4),))
    idps_rem = idps.remove(ps1)
    idps_expected = InterDependencies_(inferences=(Tree(ps2, ps4),),
                                       dependencies=(Tree(ps3),))
    assert idps_rem == idps_expected

    for p in [ps4, ps2, ps3]:
        match = re.escape(f'Cannot remove {p.name}, other parameters')
        with pytest.raises(ValueError, match=match):
            idps_rem = idps.remove(p)

    idps = InterDependencies_(dependencies=(Tree(ps1, ps3),),
                              inferences=(Tree(ps2, ps4),))
    idps_rem = idps.remove(ps2)
    idps_expected = InterDependencies_(dependencies=(Tree(ps1, ps3),
                                                     Tree(ps4)))

    assert idps_rem == idps_expected

    idps = InterDependencies_(dependencies=(Tree(ps1, ps2, ps3),
                                            Tree(ps4)))
    idps_rem = idps.remove(ps4)
    idps_expected = InterDependencies_(dependencies=(Tree(ps1, ps2, ps3),))
    assert idps_rem == idps_expected

    idps = InterDependencies_(dependencies=(Tree(ps1, ps2, ps3), Tree(ps4)))

    idps_rem = idps.remove(ps1)
    idps_expected = InterDependencies_(dependencies=(Tree(ps4),
                                                     Tree(ps2),
                                                     Tree(ps3)))
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
