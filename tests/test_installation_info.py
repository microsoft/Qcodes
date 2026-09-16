from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from packaging import version

from qcodes.utils import (
    convert_legacy_version_to_supported_version,
    get_all_installed_package_versions,
    is_qcodes_installed_editably,
)

if TYPE_CHECKING:
    import pytest
    from pytest_mock import MockerFixture

# The get_* functions from installation_info are hard to meaningfully test,
# but we can at least test that they execute without errors


def test_is_qcodes_installed_editably() -> None:
    answer = is_qcodes_installed_editably()

    assert isinstance(answer, bool)


def test_is_qcodes_installed_editably_when_pip_fails(
    caplog: pytest.LogCaptureFixture, mocker: MockerFixture
) -> None:
    """If pip cannot be queried we return None and log the traceback."""
    mocker.patch(
        "qcodes.utils.installation_info.subprocess.run",
        side_effect=OSError("pip could not be executed"),
    )

    with caplog.at_level(logging.ERROR, logger="qcodes.utils.installation_info"):
        answer = is_qcodes_installed_editably()

    assert answer is None
    assert "Could not determine if QCoDeS is installed editably" in caplog.text
    assert "pip could not be executed" in caplog.text


def test_get_all_installed_package_versions() -> None:
    ipvs = get_all_installed_package_versions()

    assert isinstance(ipvs, dict)
    assert len(ipvs) > 0

    for k, v in ipvs.items():
        assert isinstance(k, str)
        assert isinstance(v, str)


def test_convert_legacy_version_to_supported_version() -> None:
    def assert_version_str(legacy_verstr: str, expected_converted_ver_str: str) -> None:
        converted_version_str = convert_legacy_version_to_supported_version(
            legacy_verstr
        )
        assert converted_version_str == expected_converted_ver_str
        assert version.parse(converted_version_str) == version.parse(
            expected_converted_ver_str
        )

    legacy_verstr = "a.1.4"
    expected_converted_ver_str = "65.1.4"
    assert_version_str(legacy_verstr, expected_converted_ver_str)

    legacy_verstr = "10.4.7"
    expected_converted_ver_str = "10.4.7"
    assert_version_str(legacy_verstr, expected_converted_ver_str)

    legacy_verstr = "C.2.1"
    expected_converted_ver_str = "67.2.1"
    assert_version_str(legacy_verstr, expected_converted_ver_str)

    legacy_verstr = "A.02.17-02.40-02.17-00.52-04-01"
    expected_converted_ver_str = "65.02.17"
    assert_version_str(legacy_verstr, expected_converted_ver_str)
