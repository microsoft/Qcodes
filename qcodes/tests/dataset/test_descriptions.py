import json

import pytest

from qcodes.dataset.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.descriptions import RunDescriber
from qcodes.utils.helpers import YAML
from qcodes.dataset.dependencies import (InterDependencies, old_to_new,
                                         InterDependencies_, new_to_old)

from qcodes.tests.dataset.interdeps_fixtures import (some_paramspecs,
                                                     some_paramspecbases,
                                                     some_interdeps)


def test_wrong_input_type_raises():

    for interdeps in ['interdeps', ['p1', 'p2'], 0]:

        with pytest.raises(ValueError):
            RunDescriber(interdeps=interdeps)


def test_equality(some_paramspecbases):

    (psb1, psb2, psb3, psb4) = some_paramspecbases

    idp1 = InterDependencies_(dependencies={psb1: (psb2, psb3)})
    idp2 = InterDependencies_(inferences={psb1: (psb2, psb3)})
    idp3 = InterDependencies_(dependencies={psb1: (psb2, psb3)})

    desc_1 = RunDescriber(interdeps=idp1)
    desc_2 = RunDescriber(interdeps=idp2)
    desc_3 = RunDescriber(interdeps=idp3)

    assert desc_1 == desc_3
    assert desc_1 != desc_2
    assert desc_3 != desc_2


def test_serialization_dict_keys(some_interdeps):

    for idps in some_interdeps:
        desc = RunDescriber(interdeps=idps)

        ser_desc = desc.serialize()
        assert list(ser_desc.keys()) == ['version', 'interdependencies']


@pytest.mark.parametrize('version', [None, 0, 1])
def test_serialization_and_back(some_interdeps, version):

    for idps in some_interdeps:
        desc = RunDescriber(interdeps=idps)

        ser_desc = desc.serialize(version=version)

        new_desc = RunDescriber.deserialize(ser_desc)

        assert isinstance(new_desc, RunDescriber)
        assert desc == new_desc


def test_yaml_creation_and_loading(some_interdeps):

    yaml = YAML()

    for idps in some_interdeps:
        desc = RunDescriber(interdeps=idps)

        yaml_str = desc.to_yaml()
        assert isinstance(yaml_str, str)
        ydict = dict(yaml.load(yaml_str))
        assert list(ydict.keys()) == ['version', 'interdependencies']

        new_desc = RunDescriber.from_yaml(yaml_str)
        assert new_desc == desc


def test_default_jsonization_as_v0(some_interdeps):
    """
    Test that a RunDescriber always json-ifies itself as an old style
    RunDescriber, even when given new style interdeps
    """
    idps_new = some_interdeps[0]
    idps_old = new_to_old(idps_new)

    new_desc = RunDescriber(idps_new)
    old_desc = json.dumps({'version': 0,
                           'interdependencies': idps_old.serialize()})

    assert new_desc.to_json() == old_desc


def test_default_serialization_as_v0(some_interdeps):
    """
    Test that a RunDescriber always serializes itself as an old style
    RunDescriber, even when given new style interdeps
    """

    idps_new = some_interdeps[0]
    idps_old = new_to_old(idps_new)

    new_desc = RunDescriber(idps_new)
    old_desc = {'version': 0, 'interdependencies': idps_old.serialize()}

    assert new_desc.serialize() == old_desc


def test_serialization_1_as_1(some_interdeps):
    """
    Test the serialization of a version 1 object as a version1 object
    """
    for idps in some_interdeps:
        desc = RunDescriber(idps)

        ser = desc.serialize(version=1)
        assert ser['version'] == 1
        assert ser['interdependencies'] == idps.serialize()
        assert len(ser.keys()) == 2
