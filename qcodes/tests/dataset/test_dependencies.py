import pytest

from qcodes.dataset.dependencies import InterDependencies
from qcodes.dataset.param_spec import ParamSpec
from qcodes.tests.dataset.test_descriptions import some_paramspecs


def test_wrong_input_raises():

    for pspecs in [['p1', 'p2', 'p3'],
                   [ParamSpec('p1', paramtype='numeric'), 'p2'],
                   ['p1', ParamSpec('p2', paramtype='text')]]:

        with pytest.raises(ValueError):
            InterDependencies(pspecs)


def test_are_dependencies_met(some_paramspecs):

    ps1 = some_paramspecs[1]['ps1']
    ps2 = some_paramspecs[1]['ps2']
    ps3 = some_paramspecs[1]['ps3']
    ps4 = some_paramspecs[1]['ps4']
    ps5 = some_paramspecs[1]['ps5']
    ps6 = some_paramspecs[1]['ps6']

    adm = InterDependencies._are_dependencies_met

    assert adm(ps1)
    assert adm(ps1, ps2)
    assert adm(ps2, ps1)
    assert adm(ps1, ps3)
    assert adm(ps1, ps3, ps2)
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
