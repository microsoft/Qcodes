from packaging import version

from qcodes.utils import (
    convert_legacy_version_to_supported_version,
    get_all_installed_package_versions,
    is_qcodes_installed_editably,
)

# The get_* functions from installation_info are hard to meaningfully test,
# but we can at least test that they execute without errors


def test_is_qcodes_installed_editably() -> None:
    answer = is_qcodes_installed_editably()

    assert isinstance(answer, bool)


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
