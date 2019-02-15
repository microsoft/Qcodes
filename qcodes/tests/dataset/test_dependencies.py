import pytest

from qcodes.dataset.dependencies import (InterDependencies,
                                         InterDependencies_,
                                         old_to_new)
from qcodes.dataset.param_spec import ParamSpec, ParamSpecBase
from qcodes.tests.common import error_caused_by
from qcodes.tests.dataset.test_descriptions import some_paramspecs


@pytest.fixture
def some_paramspecbases():

    psb1 = ParamSpecBase('psb1', paramtype='text', label='blah', unit='')
    psb2 = ParamSpecBase('psb2', paramtype='array', label='', unit='V')

    return (psb1, psb2)


def test_wrong_input_raises():

    for pspecs in [['p1', 'p2', 'p3'],
                   [ParamSpec('p1', paramtype='numeric'), 'p2'],
                   ['p1', ParamSpec('p2', paramtype='text')]]:

        with pytest.raises(ValueError):
            InterDependencies(pspecs)


def test_init_validation_raises(some_paramspecbases):

    (ps1, ps2) = some_paramspecbases

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

    idps_old = InterDependencies(ps2, ps4, ps1, ps2, ps3, ps5, ps6)

    idps_new = old_to_new(idps_old)

    assert idps_new.dependencies == {ps5_base: (ps3_base, ps4_base),
                                     ps6_base: (ps3_base, ps4_base)}
    assert idps_new.inferences == {ps3_base: (ps1_base,),
                                   ps4_base: (ps2_base,)}
    assert idps_new.standalones == set()

    idps_old = InterDependencies(ps1, ps2)

    idps_new = old_to_new(idps_old)

    assert idps_new.dependencies == {}
    assert idps_new.inferences == {}
    assert idps_new.standalones == set((ps1_base, ps2_base))
