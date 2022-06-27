import pytest

import qcodes as qc
import qcodes.utils.installation_info as ii
from qcodes.utils import (
    QCoDeSDeprecationWarning,
    convert_legacy_version_to_supported_version,
    get_all_installed_package_versions,
    is_qcodes_installed_editably,
)

# The get_* functions from installation_info are hard to meaningfully test,
# but we can at least test that they execute without errors


def test_get_qcodes_version():
    with pytest.warns(QCoDeSDeprecationWarning):
        version = ii.get_qcodes_version()

    assert version == qc.__version__


def test_is_qcodes_installed_editably():
    answer = is_qcodes_installed_editably()

    assert isinstance(answer, bool)


def test_get_all_installed_package_versions():
    ipvs = get_all_installed_package_versions()

    assert isinstance(ipvs, dict)
    assert len(ipvs) > 0

    for k, v in ipvs.items():
        assert isinstance(k, str)
        assert isinstance(v, str)


def test_convert_legacy_version_to_supported_version():
    legacy_verstr = "a.1.4"
    assert convert_legacy_version_to_supported_version(legacy_verstr) == "65.1.4"

    legacy_verstr = "10.4.7"
    assert convert_legacy_version_to_supported_version(legacy_verstr) == "10.4.7"

    legacy_verstr = "C.2.1"
    assert convert_legacy_version_to_supported_version(legacy_verstr) == "67.2.1"
