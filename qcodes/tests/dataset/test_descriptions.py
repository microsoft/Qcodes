import json

import pytest
from ruamel.yaml import YAML

from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning import serialization as serial
from qcodes.dataset.descriptions.versioning.converters import new_to_old


def test_wrong_input_type_raises() -> None:
    for interdeps in ("interdeps", ["p1", "p2"], 0):
        with pytest.raises(ValueError):
            RunDescriber(interdeps=interdeps)  # type: ignore[arg-type]


def test_equality(some_paramspecbases) -> None:

    (psb1, psb2, psb3, _) = some_paramspecbases

    idp1 = InterDependencies_(dependencies={psb1: (psb2, psb3)})
    idp2 = InterDependencies_(inferences={psb1: (psb2, psb3)})
    idp3 = InterDependencies_(dependencies={psb1: (psb2, psb3)})

    desc_1 = RunDescriber(interdeps=idp1)
    desc_2 = RunDescriber(interdeps=idp2)
    desc_3 = RunDescriber(interdeps=idp3)

    assert desc_1 == desc_3
    assert desc_1 != desc_2
    assert desc_3 != desc_2


def test_keys_of_result_of_to_dict(some_interdeps) -> None:

    for idps in some_interdeps:
        desc = RunDescriber(interdeps=idps)

        ser_desc = desc._to_dict()
        assert list(ser_desc.keys()) == ['version',
                                         'interdependencies',
                                         'interdependencies_',
                                         'shapes']


def test_to_and_from_dict_roundtrip(some_interdeps) -> None:

    for idps in some_interdeps:
        desc = RunDescriber(interdeps=idps)

        ser_desc = desc._to_dict()

        new_desc = RunDescriber._from_dict(ser_desc)

        assert isinstance(new_desc, RunDescriber)
        assert desc == new_desc


def test_yaml_creation_and_loading(some_interdeps) -> None:

    yaml = YAML()

    for idps in some_interdeps:
        desc = RunDescriber(interdeps=idps)

        yaml_str = serial.to_yaml_for_storage(desc)
        assert isinstance(yaml_str, str)
        ydict = dict(yaml.load(yaml_str))
        assert list(ydict.keys()) == ['version',
                                      'interdependencies',
                                      'interdependencies_',
                                      'shapes']
        assert ydict['version'] == serial.STORAGE_VERSION

        new_desc = serial.from_yaml_to_current(yaml_str)
        assert new_desc == desc


def test_jsonization_as_v0_for_storage(some_interdeps) -> None:
    """
    Test that a RunDescriber can be json-dumped as version 0
    """
    idps_new = some_interdeps[0]
    idps_old = new_to_old(idps_new)

    new_desc = RunDescriber(idps_new)
    old_json = json.dumps({'version': 0,
                           'interdependencies': idps_old._to_dict()})

    assert serial.to_json_as_version(new_desc, 0) == old_json


def test_default_jsonization_for_storage(some_interdeps) -> None:
    """
    Test that a RunDescriber is json-dumped as version 2
    """
    idps_new = some_interdeps[0]
    idps_old = new_to_old(idps_new)

    new_desc = RunDescriber(idps_new)
    expected_json = json.dumps({'version': 3,
                                'interdependencies': idps_old._to_dict(),
                                'interdependencies_': idps_new._to_dict(),
                                'shapes': None})

    assert serial.to_json_for_storage(new_desc) == expected_json


def test_dictization_as_v0_for_storage(some_interdeps) -> None:
    """
    Test that a RunDescriber always gets converted to dict that represents
    an old style RunDescriber, even when given new style interdeps
    """

    idps_new = some_interdeps[0]
    idps_old = new_to_old(idps_new)

    new_desc = RunDescriber(idps_new)
    old_desc = {'version': 0, 'interdependencies': idps_old._to_dict()}

    assert serial.to_dict_as_version(new_desc, 0) == old_desc


def test_default_dictization_for_storage(some_interdeps) -> None:
    """
    Test that a RunDescriber always gets converted to dict that represents
    an old style RunDescriber, even when given new style interdeps
    """

    idps_new = some_interdeps[0]
    idps_old = new_to_old(idps_new)

    new_desc = RunDescriber(idps_new)
    old_desc = {'version': 3,
                'interdependencies': idps_old._to_dict(),
                'interdependencies_': idps_new._to_dict(),
                'shapes': None}

    assert serial.to_dict_for_storage(new_desc) == old_desc


def test_dictization_of_current_version(some_interdeps) -> None:
    """
    Test conversion to dictionary of a RunDescriber
    """
    for idps in some_interdeps:
        desc = RunDescriber(idps)
        idps_old = new_to_old(desc.interdeps)

        ser = desc._to_dict()
        assert ser['version'] == 3
        assert ser['interdependencies'] == idps_old._to_dict()
        assert ser['interdependencies_'] == idps._to_dict()
        assert ser['shapes'] is None
        assert len(ser.keys()) == 4
