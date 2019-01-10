import pytest

from qcodes.dataset.dependencies import (InterDependencies,
                                         UnknownParameterError,
                                         MissingDependencyError,
                                         DuplicateParameterError,
                                         NestedDependencyError,
                                         NestedInferenceError)
from qcodes.dataset.param_spec import ParamSpec
from qcodes.tests.dataset.test_descriptions import some_paramspecs


def test_wrong_input_raises():

    for pspecs in [['p1', 'p2', 'p3'],
                   [ParamSpec('p1', paramtype='numeric'), 'p2'],
                   ['p1', ParamSpec('p2', paramtype='text')]]:

        with pytest.raises(ValueError):
            InterDependencies(pspecs)


def test_are_dependencies_met(some_paramspecs):
    """
    note: _are_dependencies_met allows for multiple occurences of a parameter
    """

    ps1 = some_paramspecs[1]['ps1']
    ps2 = some_paramspecs[1]['ps2']
    ps3 = some_paramspecs[1]['ps3']
    ps4 = some_paramspecs[1]['ps4']
    ps5 = some_paramspecs[1]['ps5']
    ps6 = some_paramspecs[1]['ps6']

    adm = InterDependencies._are_dependencies_met

    assert adm(ps1)
    assert adm(ps1, ps1)
    assert adm(ps1, ps2)
    assert adm(ps2, ps1)
    assert adm(ps1, ps3)
    assert adm(ps1, ps3, ps2)
    assert adm(ps3, ps1, ps3, ps2, ps3)
    assert not adm(ps3)
    assert not adm(ps3, ps2)
    assert not adm(ps4, ps1)
    assert not adm(ps1, ps4)
    assert adm(ps3, ps1)
    assert adm(ps4, ps2)
    assert adm(ps5, ps4, ps3, ps2, ps1)
    assert not adm(ps5, ps4, ps3, ps1)
    assert not adm(ps5, ps4, ps3, ps2)
    assert not adm(ps5, ps4, ps2, ps1)
    assert not adm(ps5, ps3, ps2, ps1)
    assert not adm(ps6, ps3, ps4)
    assert adm(ps1, ps6, ps5, ps3, ps2, ps4)


def test_validate_dependency_levels(some_paramspecs):

    # A valid group

    ps1 = some_paramspecs[1]['ps1']
    ps2 = some_paramspecs[1]['ps2']
    ps3 = some_paramspecs[1]['ps3']
    ps4 = some_paramspecs[1]['ps4']
    ps5 = some_paramspecs[1]['ps5']
    ps6 = some_paramspecs[1]['ps6']

    vdl = InterDependencies._validate_dependency_levels

    vdl()
    vdl(ps1)
    vdl(ps1, ps1)
    vdl(ps1, ps2, ps3, ps4)
    vdl(ps1, ps3)
    vdl(ps1, ps3, ps2)
    vdl(ps5, ps4, ps3, ps2, ps1)
    vdl(ps1, ps6, ps5, ps3, ps2, ps4)
    vdl(ps1, ps6, ps5, ps3, ps2, ps4, ps1, ps2)

    # An invalid group

    ps1 = some_paramspecs[3]['ps1']
    ps2 = some_paramspecs[3]['ps2']
    ps3 = some_paramspecs[3]['ps3']
    ps4 = some_paramspecs[3]['ps4']
    ps5 = some_paramspecs[3]['ps5']
    ps6 = some_paramspecs[3]['ps6']

    vdl(ps1, ps2)
    vdl(ps4, ps5)

    with pytest.raises(NestedInferenceError):
        vdl(ps1, ps2, ps3)

    with pytest.raises(MissingDependencyError):
        vdl(ps3, ps2)

    with pytest.raises(MissingDependencyError):
        vdl(ps6, ps4)

    with pytest.raises(NestedDependencyError):
        vdl(ps4, ps5, ps6)

def test_validate_subset(some_paramspecs):

    ps1 = some_paramspecs[1]['ps1']
    ps2 = some_paramspecs[1]['ps2']
    ps3 = some_paramspecs[1]['ps3']
    ps4 = some_paramspecs[1]['ps4']
    ps5 = some_paramspecs[1]['ps5']
    ps6 = some_paramspecs[1]['ps6']

    idps = InterDependencies(*some_paramspecs[1].values())

    idps.validate_subset()

    with pytest.raises(ValueError):
        idps.validate_subset(None)

    idps.validate_subset(ps1)
    idps.validate_subset(ps3, ps2, ps1)
    idps.validate_subset(ps3, 'ps2', 'ps1')

    with pytest.raises(MissingDependencyError):
        idps.validate_subset(ps3, ps2)

    with pytest.raises(MissingDependencyError):
        idps.validate_subset('ps3', 'ps2')

    with pytest.raises(MissingDependencyError):
        idps.validate_subset(ps4, 'ps1')

    with pytest.raises(MissingDependencyError):
        idps.validate_subset('ps4', ps1)

    with pytest.raises(UnknownParameterError):
        idps.validate_subset(ps1, ps2, ps3, 'junk_parameter')

    with pytest.raises(UnknownParameterError):
        idps.validate_subset(some_paramspecs[2]['ps1'])

    with pytest.raises(DuplicateParameterError):
        idps.validate_subset(ps1, ps2, ps3, ps1)

    with pytest.raises(DuplicateParameterError):
        idps.validate_subset(ps1, ps2, ps3, 'ps1')

    idps.validate_subset(ps1, ps6, ps5, ps3, ps2, ps4)


def test_validation_on_init(some_paramspecs):
    """
    Test the self-validation at __init__
    """

    ps1 = some_paramspecs[3]['ps1']
    ps2 = some_paramspecs[3]['ps2']
    ps3 = some_paramspecs[3]['ps3']
    ps4 = some_paramspecs[3]['ps4']
    ps5 = some_paramspecs[3]['ps5']
    ps6 = some_paramspecs[3]['ps6']

    with pytest.raises(NestedInferenceError):
        InterDependencies(*some_paramspecs[3].values())

    with pytest.raises(NestedInferenceError):
        InterDependencies(ps1, ps2, ps3)

    with pytest.raises(MissingDependencyError):
        InterDependencies(ps3, ps2)

    with pytest.raises(MissingDependencyError):
        InterDependencies(ps6, ps4)

    with pytest.raises(NestedDependencyError):
        InterDependencies(ps4, ps5, ps6)

    InterDependencies()

    InterDependencies(*some_paramspecs[1].values())

    InterDependencies(some_paramspecs[1]['ps1'])

    InterDependencies(*some_paramspecs[2].values())

    with pytest.raises(NestedInferenceError):
        InterDependencies(*some_paramspecs[3].values())


