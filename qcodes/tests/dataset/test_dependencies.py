import pytest

from qcodes.dataset.dependencies import (InterDependencies,
                                         InterDependencies_)
from qcodes.dataset.param_spec import ParamSpec, ParamSpecBase
from qcodes.tests.common import error_caused_by


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
