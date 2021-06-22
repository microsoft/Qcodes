from typing import Generator

import pytest

from qcodes.dataset.export_config import DataExportType
from qcodes.dataset.exporters.export_info import ExportInfo


@pytest.fixture(name="basic_export_info")
def _make_basic_export_info() -> Generator[ExportInfo, None, None]:
    nc_path = "D:\\data\\33.nc"
    csv_path = "D:\\data\\33.csv"

    export_info = ExportInfo(
        {DataExportType.NETCDF: nc_path, DataExportType.CSV: csv_path}
    )
    yield export_info


def test_export_info_basic() -> None:
    nc_path = "D:\\data\\33.nc"
    csv_path = "D:\\data\\33.csv"

    a = ExportInfo({DataExportType.NETCDF: nc_path, DataExportType.CSV: csv_path})
    assert a.export_paths[DataExportType.NETCDF] == nc_path
    assert a.export_paths[DataExportType.CSV] == csv_path


def test_export_info_json_roundtrip(basic_export_info) -> None:

    loaded_export_info = basic_export_info.to_str()
    1 + 1
    # assert loaded_export_info == basic_export_info
