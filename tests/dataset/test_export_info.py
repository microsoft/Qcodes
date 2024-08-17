from typing import TYPE_CHECKING

import pytest

from qcodes.dataset.export_config import DataExportType
from qcodes.dataset.exporters.export_info import ExportInfo

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(name="basic_export_info")
def _make_basic_export_info() -> "Generator[ExportInfo, None, None]":
    nc_path = "D:\\data\\33.nc"
    csv_path = "D:\\data\\33.csv"

    export_info = ExportInfo(
        {DataExportType.NETCDF.value: nc_path, DataExportType.CSV.value: csv_path}
    )
    yield export_info


def test_export_info_basic() -> None:
    nc_path = "D:\\data\\33.nc"
    csv_path = "D:\\data\\33.csv"

    a = ExportInfo(
        {DataExportType.NETCDF.value: nc_path, DataExportType.CSV.value: csv_path}
    )
    assert a.export_paths[DataExportType.NETCDF.value] == nc_path
    assert a.export_paths[DataExportType.CSV.value] == csv_path


def test_invalid_key_raises() -> None:
    nd_path = "D:\\data\\33.nd"

    with pytest.warns(Warning, match="The supported export types are"):
        _ = ExportInfo({"nd": nd_path})


def test_export_info_json_roundtrip(basic_export_info) -> None:
    exported_str = basic_export_info.to_str()
    loaded_export_info = ExportInfo.from_str(exported_str)
    assert loaded_export_info == basic_export_info


def test_init_from_empty_str() -> None:
    export_info = ExportInfo.from_str("")
    assert export_info.export_paths == {}
