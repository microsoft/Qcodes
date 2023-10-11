from deepdiff import DeepDiff  # type: ignore[import-untyped]

from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning.converters import (
    new_to_old,
    old_to_new,
    v0_to_v1,
    v0_to_v2,
    v1_to_v0,
    v1_to_v2,
    v2_to_v0,
    v2_to_v1,
)
from qcodes.dataset.descriptions.versioning.rundescribertypes import (
    RunDescriberV0Dict,
    RunDescriberV1Dict,
    RunDescriberV2Dict,
    RunDescriberV3Dict,
)
from qcodes.dataset.descriptions.versioning.serialization import from_dict_to_current
from qcodes.dataset.descriptions.versioning.v0 import InterDependencies


def test_convert_v0_to_newer(some_paramspecs) -> None:
    pgroup1 = some_paramspecs[1]

    interdeps = InterDependencies(pgroup1['ps1'],
                                  pgroup1['ps2'],
                                  pgroup1['ps3'],
                                  pgroup1['ps4'],
                                  pgroup1['ps6'])

    v0 = RunDescriberV0Dict(interdependencies=interdeps._to_dict(), version=0)
    v1 = v0_to_v1(v0)
    v2 = v0_to_v2(v0)

    _assert_dicts_are_related_as_expected(v0, v1, v2)


def test_convert_v1(some_interdeps) -> None:
    interdeps_ = some_interdeps[0]

    v1 = RunDescriberV1Dict(interdependencies=interdeps_._to_dict(),
                            version=1)
    v0 = v1_to_v0(v1)
    v2 = v1_to_v2(v1)
    _assert_dicts_are_related_as_expected(v0, v1, v2)


def test_convert_v2(some_interdeps) -> None:
    interdeps_ = some_interdeps[0]
    interdeps = new_to_old(interdeps_)

    v2 = RunDescriberV2Dict(interdependencies=interdeps._to_dict(),
                            interdependencies_=interdeps_._to_dict(),
                            version=2)
    v1 = v2_to_v1(v2)
    v0 = v2_to_v0(v2)
    _assert_dicts_are_related_as_expected(v0, v1, v2)


def _assert_dicts_are_related_as_expected(v0, v1, v2):
    assert v1['interdependencies'] == old_to_new(
        InterDependencies._from_dict(v0['interdependencies'])
    )._to_dict()
    assert v1['version'] == 1
    assert len(v1) == 2

    # conversion does not preserve order in the dict so use deepdiff to compare
    assert DeepDiff(v2['interdependencies'], v0['interdependencies'],
                    ignore_order=True) == {}
    assert v2['interdependencies_'] == v1['interdependencies']
    assert v2['version'] == 2
    assert len(v2) == 3


def test_construct_current_rundescriber_from_v0(some_paramspecs) -> None:

    pgroup1 = some_paramspecs[1]

    interdeps = InterDependencies(pgroup1['ps1'],
                                  pgroup1['ps2'],
                                  pgroup1['ps3'],
                                  pgroup1['ps4'],
                                  pgroup1['ps6'])
    v0 = RunDescriberV0Dict(interdependencies=interdeps._to_dict(), version=0)
    rds1 = RunDescriber._from_dict(v0)

    rds_upgraded = from_dict_to_current(v0)

    expected_v3_dict = RunDescriberV3Dict(
        interdependencies=interdeps._to_dict(),
        interdependencies_=old_to_new(interdeps)._to_dict(),
        version=3,
        shapes=None,
    )
    assert DeepDiff(rds1._to_dict(), expected_v3_dict,
                    ignore_order=True) == {}
    assert DeepDiff(rds_upgraded._to_dict(), expected_v3_dict,
                    ignore_order=True) == {}


def test_construct_current_rundescriber_from_v1(some_interdeps) -> None:
    interdeps_ = some_interdeps[0]
    interdeps = new_to_old(interdeps_)

    v1 = RunDescriberV1Dict(interdependencies=interdeps_._to_dict(),
                            version=1)
    rds1 = RunDescriber._from_dict(v1)
    rds_upgraded = from_dict_to_current(v1)

    expected_v3_dict = RunDescriberV3Dict(
        interdependencies=interdeps._to_dict(),
        interdependencies_=interdeps_._to_dict(),
        version=3,
        shapes=None,
    )
    assert rds1._to_dict() == expected_v3_dict
    assert rds_upgraded._to_dict() == expected_v3_dict


def test_construct_current_rundescriber_from_v2(some_interdeps) -> None:
    interdeps_ = some_interdeps[0]
    interdeps = new_to_old(interdeps_)

    v2 = RunDescriberV2Dict(interdependencies=interdeps._to_dict(),
                            interdependencies_=interdeps_._to_dict(),
                            version=2)

    expected_v3_dict = RunDescriberV3Dict(
        interdependencies=interdeps._to_dict(),
        interdependencies_=interdeps_._to_dict(),
        version=3,
        shapes=None,
    )
    rds1 = RunDescriber._from_dict(v2)
    rds_upgraded = from_dict_to_current(v2)

    assert rds1._to_dict() == expected_v3_dict
    assert rds_upgraded._to_dict() == expected_v3_dict


def test_construct_current_rundescriber_from_v3(some_interdeps) -> None:
    interdeps_ = some_interdeps[0]
    interdeps = new_to_old(interdeps_)

    v3 = RunDescriberV3Dict(interdependencies=interdeps._to_dict(),
                            interdependencies_=interdeps_._to_dict(),
                            version=3,
                            shapes=None)
    rds1 = RunDescriber._from_dict(v3)
    rds_upgraded = from_dict_to_current(v3)
    assert rds1._to_dict() == v3
    assert rds_upgraded._to_dict() == v3


def test_construct_current_rundescriber_from_fake_v4(some_interdeps) -> None:
    interdeps_ = some_interdeps[0]
    interdeps = new_to_old(interdeps_)

    v4 = RunDescriberV3Dict(
        interdependencies=interdeps._to_dict(),
        interdependencies_=interdeps_._to_dict(),
        version=4,
        shapes=None,
    )
    v4["foobar"] = {"foo": ["bar"]}  # type: ignore[typeddict-unknown-key]
    rds1 = RunDescriber._from_dict(v4)
    rds_upgraded = from_dict_to_current(v4)
    v3 = v4.copy()
    v3.pop("foobar")  # type: ignore[typeddict-item]
    v3["version"] = 3
    assert rds1._to_dict() == v3
    assert rds_upgraded._to_dict() == v3
