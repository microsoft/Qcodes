from __future__ import annotations

from qcodes.dataset.export_config import (
    DataExportType,
    get_data_export_name_elements,
    get_data_export_prefix,
    get_data_export_type,
    set_data_export_prefix,
    set_data_export_type,
)


def test_data_export_type_enum_members() -> None:
    assert DataExportType.NETCDF.value == "nc"
    assert DataExportType.CSV.value == "csv"
    assert len(DataExportType) == 2


def test_get_data_export_type_with_string_netcdf() -> None:
    result = get_data_export_type("NETCDF")
    assert result is DataExportType.NETCDF


def test_get_data_export_type_with_string_csv() -> None:
    result = get_data_export_type("CSV")
    assert result is DataExportType.CSV


def test_get_data_export_type_case_insensitive() -> None:
    assert get_data_export_type("netcdf") is DataExportType.NETCDF
    assert get_data_export_type("csv") is DataExportType.CSV
    assert get_data_export_type("Csv") is DataExportType.CSV


def test_get_data_export_type_with_enum_input() -> None:
    result = get_data_export_type(DataExportType.NETCDF)
    assert result is DataExportType.NETCDF

    result = get_data_export_type(DataExportType.CSV)
    assert result is DataExportType.CSV


def test_get_data_export_type_with_none_returns_none() -> None:
    # When config export_type is also None/empty, should return None
    set_data_export_type(None)  # type: ignore[arg-type]
    result = get_data_export_type(None)
    assert result is None


def test_get_data_export_type_with_invalid_string_returns_none() -> None:
    result = get_data_export_type("nonexistent_format")
    assert result is None


def test_set_and_get_data_export_prefix_roundtrip() -> None:
    set_data_export_prefix("my_prefix_")
    assert get_data_export_prefix() == "my_prefix_"

    set_data_export_prefix("")
    assert get_data_export_prefix() == ""


def test_get_data_export_name_elements_returns_list() -> None:
    result = get_data_export_name_elements()
    assert isinstance(result, list)


def test_set_data_export_type_valid() -> None:
    set_data_export_type("netcdf")
    result = get_data_export_type()
    assert result is DataExportType.NETCDF

    set_data_export_type("csv")
    result = get_data_export_type()
    assert result is DataExportType.CSV


def test_set_data_export_type_invalid_does_not_change_config() -> None:
    set_data_export_type("netcdf")
    set_data_export_type("invalid_type")
    # Config should still have the previous valid value
    result = get_data_export_type()
    assert result is DataExportType.NETCDF
