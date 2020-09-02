import pytest
from deepdiff import DeepDiff

from qcodes.dataset.descriptions.versioning.v0 import InterDependencies
from qcodes.dataset.descriptions.versioning.rundescribertypes import (RunDescriberV0Dict,
                                                                      RunDescriberV1Dict,
                                                                      RunDescriberV2Dict)
from qcodes.dataset.descriptions.versioning.converters import (v0_to_v1, v0_to_v2, v1_to_v0, v1_to_v2,
                                                               v2_to_v0, v2_to_v1, old_to_new, new_to_old)


def test_convert_v0_to_newer(some_paramspecs):
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


def test_convert_v1(some_interdeps):
    interdeps_ = some_interdeps[0]

    v1 = RunDescriberV1Dict(interdependencies=interdeps_._to_dict(),
                            version=1)
    v0 = v1_to_v0(v1)
    v2 = v1_to_v2(v1)
    _assert_dicts_are_related_as_expected(v0, v1, v2)


def test_convert_v2(some_interdeps):
    interdeps_ = some_interdeps[0]
    interdeps = new_to_old(interdeps_)

    v2 = RunDescriberV2Dict(interdependencies=interdeps._to_dict(),
                            interdependencies_=interdeps_._to_dict(),
                            version=2)
    v1 = v2_to_v1(v2)
    v0 = v2_to_v0(v2)
    _assert_dicts_are_related_as_expected(v0, v1, v2)


def _assert_dicts_are_related_as_expected(v0, v1, v2):
    assert v1['interdependencies'] == old_to_new(InterDependencies._from_dict(v0['interdependencies']))._to_dict()
    assert v1['version'] == 1
    assert len(v1) == 2

    # conversion does not preserve order in the dict so use deepdiff to compare
    assert DeepDiff(v2['interdependencies'], v0['interdependencies'], ignore_order=True) == {}
    assert v2['interdependencies_'] == v1['interdependencies']
    assert v2['version'] == 2
    assert len(v2) == 3
