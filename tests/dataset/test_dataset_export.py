from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as hst
from numpy.testing import assert_allclose
from pytest import LogCaptureFixture, TempPathFactory

import qcodes
from qcodes.dataset import (
    DataSetProtocol,
    DataSetType,
    Measurement,
    get_data_export_path,
    load_by_guid,
    load_by_id,
    load_from_netcdf,
    new_data_set,
)
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.descriptions.versioning import serialization as serial
from qcodes.dataset.export_config import DataExportType
from qcodes.dataset.exporters.export_to_pandas import _generate_pandas_index
from qcodes.dataset.exporters.export_to_xarray import _calculate_index_shape
from qcodes.dataset.linked_datasets.links import links_to_str

if TYPE_CHECKING:
    from qcodes.dataset.data_set import DataSet


@pytest.fixture(name="mock_empty_dataset")
def _make_mock_empty_dataset(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric")
    yparam = ParamSpecBase("y", "numeric")
    zparam = ParamSpecBase("z", "numeric")
    idps = InterDependencies_(dependencies={yparam: (xparam,), zparam: (xparam,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset")
def _make_mock_dataset(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric")
    yparam = ParamSpecBase("y", "numeric")
    zparam = ParamSpecBase("z", "numeric")
    idps = InterDependencies_(dependencies={yparam: (xparam,), zparam: (xparam,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    results = [{"x": 0, "y": 1, "z": 2}]
    dataset.add_results(results)
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_nonunique")
def _make_mock_dataset_nonunique_index(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric")
    yparam = ParamSpecBase("y", "numeric")
    zparam = ParamSpecBase("z", "numeric")
    idps = InterDependencies_(dependencies={yparam: (xparam,), zparam: (xparam,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    results = [{"x": 0, "y": 1, "z": 2}, {"x": 0, "y": 1, "z": 2}]
    dataset.add_results(results)
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_label_unit")
def _make_mock_dataset_label_unit(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric", label="x label", unit="x unit")
    yparam = ParamSpecBase("y", "numeric", label="y label", unit="y unit")
    zparam = ParamSpecBase("z", "numeric", label="z label", unit="z unit")
    idps = InterDependencies_(dependencies={yparam: (xparam,), zparam: (xparam,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    results = [{"x": 0, "y": 1, "z": 2}]
    dataset.add_results(results)
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_complex")
def _make_mock_dataset_complex(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric")
    yparam = ParamSpecBase("y", "complex")
    idps = InterDependencies_(dependencies={yparam: (xparam,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    results = [{"x": 0, "y": 1 + 1j}]
    dataset.add_results(results)
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_grid")
def _make_mock_dataset_grid(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric")
    yparam = ParamSpecBase("y", "numeric")
    zparam = ParamSpecBase("z", "numeric")
    idps = InterDependencies_(dependencies={zparam: (xparam, yparam)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    for x in range(10):
        for y in range(20, 25):
            results = [{"x": x, "y": y, "z": x + y}]
            dataset.add_results(results)
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_in_mem_grid")
def _make_mock_dataset_in_mem_grid(experiment) -> DataSetProtocol:
    meas = Measurement(exp=experiment, name="in_mem_ds")
    meas.register_custom_parameter("x", paramtype="numeric")
    meas.register_custom_parameter("y", paramtype="numeric")
    meas.register_custom_parameter("z", paramtype="numeric", setpoints=("x", "y"))

    with meas.run(dataset_class=DataSetType.DataSetInMem) as datasaver:
        for x in range(10):
            for y in range(20, 25):
                results: list[tuple[str, int]] = [("x", x), ("y", y), ("z", x + y)]
                datasaver.add_result(*results)
    return datasaver.dataset


@pytest.fixture(name="mock_dataset_grid_with_shapes")
def _make_mock_dataset_grid_with_shapes(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric")
    yparam = ParamSpecBase("y", "numeric")
    zparam = ParamSpecBase("z", "numeric")
    idps = InterDependencies_(dependencies={zparam: (xparam, yparam)})
    dataset.set_interdependencies(idps, shapes={"z": (10, 5)})

    dataset.mark_started()
    for x in range(10):
        for y in range(20, 25):
            results = [{"x": x, "y": y, "z": x + y}]
            dataset.add_results(results)
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_grid_incomplete")
def _make_mock_dataset_grid_incomplete(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric")
    yparam = ParamSpecBase("y", "numeric")
    zparam = ParamSpecBase("z", "numeric")
    idps = InterDependencies_(dependencies={zparam: (xparam, yparam)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    break_loop = False

    for x in range(10):
        if break_loop is True:
            break
        for y in range(20, 25):
            results = [{"x": x, "y": y, "z": x + y}]
            dataset.add_results(results)
            if y == 23 and x == 7:
                break_loop = True
                break
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_grid_incomplete_with_shapes")
def _make_mock_dataset_grid_incomplete_with_shapes(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric")
    yparam = ParamSpecBase("y", "numeric")
    zparam = ParamSpecBase("z", "numeric")
    idps = InterDependencies_(dependencies={zparam: (xparam, yparam)})
    dataset.set_interdependencies(idps, shapes={"z": (10, 5)})

    dataset.mark_started()
    break_loop = False

    for x in range(10):
        if break_loop is True:
            break
        for y in range(20, 25):
            results = [{"x": x, "y": y, "z": x + y}]
            dataset.add_results(results)
            if y == 23 and x == 7:
                break_loop = True
                break
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_numpy")
def _make_mock_dataset_numpy(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric", label="x label", unit="x unit")
    yparam = ParamSpecBase("y", "array", label="y label", unit="y unit")
    zparam = ParamSpecBase("z", "array", label="z label", unit="z unit")
    idps = InterDependencies_(dependencies={zparam: (xparam, yparam)})
    dataset.set_interdependencies(idps)

    y = np.arange(10, 21, 1)
    dataset.mark_started()
    for x in range(10):
        results: list[dict[str, int | np.ndarray]] = [{"x": x, "y": y, "z": x + y}]
        dataset.add_results(results)
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_numpy_complex")
def _make_mock_dataset_numpy_complex(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric", label="x label", unit="x unit")
    yparam = ParamSpecBase("y", "array", label="y label", unit="y unit")
    zparam = ParamSpecBase("z", "array", label="z label", unit="z unit")
    idps = InterDependencies_(dependencies={zparam: (xparam, yparam)})
    dataset.set_interdependencies(idps)

    y = np.arange(10, 21, 1)
    dataset.mark_started()
    for x in range(10):
        results: list[dict[str, int | np.ndarray]] = [{"x": x, "y": y, "z": x + 1j * y}]
        dataset.add_results(results)
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_non_grid")
def _make_mock_dataset_non_grid(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric")
    yparam = ParamSpecBase("y", "numeric")
    zparam = ParamSpecBase("z", "numeric")
    idps = InterDependencies_(dependencies={zparam: (xparam, yparam)})
    dataset.set_interdependencies(idps)

    num_samples = 50

    rng = np.random.default_rng()

    x_vals = rng.random(num_samples) * 10
    y_vals = 20 + rng.random(num_samples) * 5

    dataset.mark_started()

    for x, y in zip(x_vals, y_vals):
        results = [{"x": x, "y": y, "z": x + y}]
        dataset.add_results(results)
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_non_grid_in_mem")
def _make_mock_dataset_non_grid_in_mem(experiment) -> DataSetProtocol:
    meas = Measurement(exp=experiment, name="in_mem_ds")

    num_samples = 50

    rng = np.random.default_rng()

    x_vals = rng.random(num_samples) * 10
    y_vals = 20 + rng.random(num_samples) * 5

    meas.register_custom_parameter("x")
    meas.register_custom_parameter("y")
    meas.register_custom_parameter("z", setpoints=("x", "y"))

    with meas.run(dataset_class=DataSetType.DataSetInMem) as datasaver:
        for x, y in zip(x_vals, y_vals):
            results = [("x", x), ("y", y), ("z", x + y)]
            datasaver.add_result(*results)

    return datasaver.dataset


@pytest.fixture(name="mock_dataset_non_grid_in_grid")
def _make_mock_dataset_non_grid_in_grid(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric")
    y1param = ParamSpecBase("y1", "numeric")
    y2param = ParamSpecBase("y2", "numeric")
    zparam = ParamSpecBase("z", "numeric")
    idps = InterDependencies_(dependencies={zparam: (xparam, y1param, y2param)})
    dataset.set_interdependencies(idps)

    num_samples = 50

    rng = np.random.default_rng()

    dataset.mark_started()
    for x in range(1, 10):
        y1_vals = rng.random(num_samples) * 10
        y2_vals = 20 + rng.random(num_samples) * 5
        for y1, y2 in zip(y1_vals, y2_vals):
            results = [{"x": x, "y1": y1, "y2": y2, "z": x + y1 + y2}]
            dataset.add_results(results)
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_grid_in_non_grid")
def _make_mock_dataset_grid_in_non_grid(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    x1param = ParamSpecBase("x1", "numeric")
    x2param = ParamSpecBase("x2", "numeric")
    yparam = ParamSpecBase("y", "numeric")
    zparam = ParamSpecBase("z", "numeric")
    idps = InterDependencies_(dependencies={zparam: (x1param, x2param, yparam)})
    dataset.set_interdependencies(idps)

    num_x_samples = 10

    rng = np.random.default_rng()

    dataset.mark_started()
    x1_vals = rng.random(num_x_samples)
    x2_vals = rng.random(num_x_samples)
    for x1, x2 in zip(x1_vals, x2_vals):
        for y in range(5):
            results = [{"x1": x1, "x2": x2, "y": y, "z": x1 + x2 + y}]
            dataset.add_results(results)
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_non_grid_in_non_grid")
def _make_mock_dataset_non_grid_in_non_grid(experiment) -> DataSet:
    dataset = new_data_set("dataset")
    x1param = ParamSpecBase("x1", "numeric")
    x2param = ParamSpecBase("x2", "numeric")
    y1param = ParamSpecBase("y1", "numeric")
    y2param = ParamSpecBase("y2", "numeric")
    zparam = ParamSpecBase("z", "numeric")
    idps = InterDependencies_(
        dependencies={zparam: (x1param, x2param, y1param, y2param)}
    )
    dataset.set_interdependencies(idps)

    num_x_samples = 10
    num_y_samples = 5

    rng = np.random.default_rng()

    dataset.mark_started()
    x1_vals = rng.random(num_x_samples)
    x2_vals = rng.random(num_x_samples)
    for x1, x2 in zip(x1_vals, x2_vals):
        y1_vals = rng.random(num_y_samples) * 10
        y2_vals = 20 + rng.random(num_y_samples) * 5
        for y1, y2 in zip(y1_vals, y2_vals):
            results = [{"x1": x1, "x2": x2, "y1": y1, "y2": y2, "z": x1 + x2 + y1 + y2}]
            dataset.add_results(results)
    dataset.mark_completed()
    return dataset


@pytest.fixture(name="mock_dataset_inverted_coords")
def _make_mock_dataset_inverted_coords(experiment) -> DataSet:
    # this dataset is constructed such
    # that the two z parameters have inverted
    # coordinates. You almost certainly
    # don't want to do this in a real dataset
    # but it enables the test to check that
    # the order is preserved.
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric")
    yparam = ParamSpecBase("y", "numeric")
    z1param = ParamSpecBase("z1", "numeric")
    z2param = ParamSpecBase("z2", "numeric")
    idps = InterDependencies_(
        dependencies={z1param: (xparam, yparam), z2param: (yparam, xparam)}
    )
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    for x in range(10):
        for y in range(20, 25):
            results = [{"x": x, "y": y, "z1": x + y, "z2": x - y}]
            dataset.add_results(results)
    dataset.mark_completed()
    return dataset


@pytest.mark.usefixtures("experiment")
def test_write_data_to_text_file_save(tmp_path_factory) -> None:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric")
    yparam = ParamSpecBase("y", "numeric")
    idps = InterDependencies_(dependencies={yparam: (xparam,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    results = [{"x": 0, "y": 1}]
    dataset.add_results(results)
    dataset.mark_completed()

    path = str(tmp_path_factory.mktemp("write_data_to_text_file_save"))
    dataset.write_data_to_text_file(path=path)
    assert os.listdir(path) == ["y.dat"]
    with open(os.path.join(path, "y.dat")) as f:
        assert f.readlines() == ["0.0\t1.0\n"]


def test_write_data_to_text_file_save_multi_keys(
    tmp_path_factory, mock_dataset
) -> None:
    tmp_path = tmp_path_factory.mktemp("data_to_text_file_save_multi_keys")
    path = str(tmp_path)
    mock_dataset.write_data_to_text_file(path=path)
    assert sorted(os.listdir(path)) == ["y.dat", "z.dat"]
    with open(os.path.join(path, "y.dat")) as f:
        assert f.readlines() == ["0.0\t1.0\n"]
    with open(os.path.join(path, "z.dat")) as f:
        assert f.readlines() == ["0.0\t2.0\n"]


def test_write_data_to_text_file_save_single_file(
    tmp_path_factory, mock_dataset
) -> None:
    tmp_path = tmp_path_factory.mktemp("to_text_file_save_single_file")
    path = str(tmp_path)
    mock_dataset.write_data_to_text_file(
        path=path, single_file=True, single_file_name="yz"
    )
    assert os.listdir(path) == ["yz.dat"]
    with open(os.path.join(path, "yz.dat")) as f:
        assert f.readlines() == ["0.0\t1.0\t2.0\n"]


@pytest.mark.usefixtures("experiment")
def test_write_data_to_text_file_length_exception(tmp_path) -> None:
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", "numeric")
    yparam = ParamSpecBase("y", "numeric")
    zparam = ParamSpecBase("z", "numeric")
    idps = InterDependencies_(dependencies={yparam: (xparam,), zparam: (xparam,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    results1 = [{"x": 0, "y": 1}]
    results2 = [{"x": 0, "z": 2}]
    results3 = [{"x": 1, "z": 3}]
    dataset.add_results(results1)
    dataset.add_results(results2)
    dataset.add_results(results3)
    dataset.mark_completed()

    temp_dir = str(tmp_path)
    with pytest.raises(Exception, match="different length"):
        dataset.write_data_to_text_file(
            path=temp_dir, single_file=True, single_file_name="yz"
        )


def test_write_data_to_text_file_name_exception(tmp_path, mock_dataset) -> None:
    temp_dir = str(tmp_path)
    with pytest.raises(Exception, match="desired file name"):
        mock_dataset.write_data_to_text_file(
            path=temp_dir, single_file=True, single_file_name=None
        )


def test_export_csv(tmp_path_factory, mock_dataset, caplog: LogCaptureFixture) -> None:
    tmp_path = tmp_path_factory.mktemp("export_csv")
    path = str(tmp_path)
    with caplog.at_level(logging.INFO):
        mock_dataset.export(export_type="csv", path=path, prefix="qcodes_")

    assert "Executing on_export callback log_exported_ds" in caplog.messages
    assert any("Dataset has been exported to" in mes for mes in caplog.messages)
    assert any("this was triggered manually" in mes for mes in caplog.messages)

    mock_dataset.add_metadata("metadata_added_after_export", 69)

    expected_path = f"qcodes_{mock_dataset.captured_run_id}_{mock_dataset.guid}.csv"
    expected_full_path = os.path.join(path, expected_path)
    assert mock_dataset.export_info.export_paths["csv"] == expected_full_path
    assert os.listdir(path) == [expected_path]
    with open(expected_full_path) as f:
        assert f.readlines() == ["0.0\t1.0\t2.0\n"]


def test_export_netcdf(
    tmp_path_factory, mock_dataset, caplog: LogCaptureFixture
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    path = str(tmp_path)
    with caplog.at_level(logging.INFO):
        mock_dataset.export(export_type="netcdf", path=path, prefix="qcodes_")

    assert "Executing on_export callback log_exported_ds" in caplog.messages
    assert any("Dataset has been exported to" in mes for mes in caplog.messages)
    assert any("this was triggered manually" in mes for mes in caplog.messages)

    mock_dataset.add_metadata("metadata_added_after_export", 69)

    expected_path = f"qcodes_{mock_dataset.captured_run_id}_{mock_dataset.guid}.nc"
    assert os.listdir(path) == [expected_path]
    file_path = os.path.join(path, expected_path)
    ds = xr.open_dataset(file_path)
    df = ds.to_dataframe()
    assert df.index.name == "x"
    assert df.index.values.tolist() == [0.0]
    assert df.y.values.tolist() == [1.0]
    assert df.z.values.tolist() == [2.0]
    expected_attrs = mock_dataset.metadata.copy()
    expected_attrs.pop("export_info")
    for attr, val in expected_attrs.items():
        assert ds.attrs[attr] == val

    assert mock_dataset.export_info.export_paths["nc"] == file_path


def test_export_netcdf_default_dir(
    tmp_path_factory: TempPathFactory, mock_dataset
) -> None:
    qcodes.config.dataset.export_path = "{db_location}"
    mock_dataset.export(export_type="netcdf", prefix="qcodes_")
    export_path = Path(mock_dataset.export_info.export_paths["nc"])
    exported_dir = export_path.parent
    export_dir_stem = exported_dir.stem
    database_path = Path(qcodes.config.core.db_location)
    database_file_name = database_path.name
    database_dir = Path(qcodes.config.core.db_location).parent
    assert qcodes.config.dataset.export_path == "{db_location}"
    assert exported_dir.parent == database_dir
    assert export_dir_stem == database_file_name.replace(".", "_")
    assert exported_dir == get_data_export_path()


def test_export_netcdf_csv(tmp_path_factory, mock_dataset) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    path = str(tmp_path)
    csv_path = os.path.join(
        path, f"qcodes_{mock_dataset.captured_run_id}_{mock_dataset.guid}.csv"
    )
    nc_path = os.path.join(
        path, f"qcodes_{mock_dataset.captured_run_id}_{mock_dataset.guid}.nc"
    )

    mock_dataset.export(export_type="netcdf", path=path, prefix="qcodes_")
    mock_dataset.export(export_type="csv", path=path, prefix="qcodes_")

    mock_dataset.add_metadata("metadata_added_after_export", 69)

    assert mock_dataset.export_info.export_paths["nc"] == nc_path
    assert mock_dataset.export_info.export_paths["csv"] == csv_path

    loaded_xr_ds = xr.open_dataset(nc_path)
    assert loaded_xr_ds.attrs["metadata_added_after_export"] == 69

    mock_dataset.export(export_type="netcdf", path=path, prefix="foobar_")
    new_nc_path = os.path.join(
        path, f"foobar_{mock_dataset.captured_run_id}_{mock_dataset.guid}.nc"
    )

    mock_dataset.add_metadata("metadata_added_after_export_2", 696)

    assert mock_dataset.export_info.export_paths["nc"] == new_nc_path
    assert mock_dataset.export_info.export_paths["csv"] == csv_path

    loaded_xr_ds = xr.open_dataset(nc_path)
    assert loaded_xr_ds.attrs["metadata_added_after_export"] == 69
    assert "metadata_added_after_export_2" not in loaded_xr_ds.attrs

    loaded_new_xr_ds = xr.open_dataset(new_nc_path)
    assert loaded_new_xr_ds.attrs["metadata_added_after_export"] == 69
    assert loaded_new_xr_ds.attrs["metadata_added_after_export_2"] == 696


def test_export_netcdf_complex_data(tmp_path_factory, mock_dataset_complex) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    path = str(tmp_path)
    mock_dataset_complex.export(export_type="netcdf", path=path, prefix="qcodes_")
    short_path = (
        f"qcodes_{mock_dataset_complex.captured_run_id}_{mock_dataset_complex.guid}.nc"
    )
    assert os.listdir(path) == [short_path]
    file_path = os.path.join(path, short_path)
    # need to explicitly use h5netcdf when reading or complex data vars will be empty
    ds = xr.open_dataset(file_path, engine="h5netcdf")
    df = ds.to_dataframe()
    assert df.index.name == "x"
    assert df.index.values.tolist() == [0.0]
    assert df.y.values.tolist() == [1.0 + 1j]


def test_export_no_or_nonexistent_type_specified(
    tmp_path_factory, mock_dataset
) -> None:
    with pytest.raises(ValueError, match="No data export type specified"):
        mock_dataset.export()

    with pytest.raises(ValueError, match="Export type foo is unknown."):
        mock_dataset.export(export_type="foo")


def test_export_from_config(tmp_path_factory, mock_dataset, mocker) -> None:
    tmp_path = tmp_path_factory.mktemp("export_from_config")
    path = str(tmp_path)
    mock_type = mocker.patch("qcodes.dataset.data_set_protocol.get_data_export_type")
    mock_path = mocker.patch("qcodes.dataset.data_set_protocol.get_data_export_path")
    mock_type.return_value = DataExportType.CSV
    mock_path.return_value = tmp_path
    mock_dataset.export()
    assert os.listdir(path) == [
        f"qcodes_{mock_dataset.captured_run_id}_{mock_dataset.guid}.csv"
    ]


def test_export_from_config_set_name_elements(
    tmp_path_factory, mock_dataset, mocker
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_from_config")
    path = str(tmp_path)
    mock_type = mocker.patch("qcodes.dataset.data_set_protocol.get_data_export_type")
    mock_path = mocker.patch("qcodes.dataset.data_set_protocol.get_data_export_path")
    mock_name_elements = mocker.patch(
        "qcodes.dataset.data_set_protocol.get_data_export_name_elements"
    )
    mock_type.return_value = DataExportType.CSV
    mock_path.return_value = tmp_path
    mock_name_elements.return_value = [
        "captured_run_id",
        "guid",
        "exp_name",
        "sample_name",
        "name",
    ]
    mock_dataset.export()
    assert os.listdir(path) == [
        f"qcodes_{mock_dataset.captured_run_id}_{mock_dataset.guid}_{mock_dataset.exp_name}_{mock_dataset.sample_name}_{mock_dataset.name}.csv"
    ]


def test_same_setpoint_warning_for_df_and_xarray(different_setpoint_dataset) -> None:
    warning_message = (
        "Independent parameter setpoints are not equal. "
        "Check concatenated output carefully."
    )

    with pytest.warns(UserWarning, match=warning_message):
        different_setpoint_dataset.to_pandas_dataframe()

    with pytest.warns(UserWarning, match=warning_message):
        different_setpoint_dataset.to_xarray_dataset()

    with pytest.warns(UserWarning, match=warning_message):
        different_setpoint_dataset.cache.to_pandas_dataframe()

    with pytest.warns(UserWarning, match=warning_message):
        different_setpoint_dataset.cache.to_xarray_dataset()


def test_export_to_xarray_dataset_empty_ds(mock_empty_dataset) -> None:
    ds = mock_empty_dataset.to_xarray_dataset()
    assert len(ds) == 2
    assert len(ds.coords) == 1
    assert "x" in ds.coords
    _assert_xarray_metadata_is_as_expected(ds, mock_empty_dataset)


def test_export_to_xarray_dataarray_empty_ds(mock_empty_dataset) -> None:
    dad = mock_empty_dataset.to_xarray_dataarray_dict()
    assert len(dad) == 2
    assert len(dad["y"].coords) == 1
    assert "x" in dad["y"].coords
    assert len(dad["z"].coords) == 1
    assert "x" in dad["z"].coords


def test_export_to_xarray(mock_dataset) -> None:
    ds = mock_dataset.to_xarray_dataset()
    assert len(ds) == 2
    assert "index" not in ds.coords
    assert "x" in ds.coords
    _assert_xarray_metadata_is_as_expected(ds, mock_dataset)


def test_export_to_xarray_non_unique_dependent_parameter(
    mock_dataset_nonunique,
) -> None:
    """When x (the dependent parameter) contains non unique values it cannot be used
    as coordinates in xarray so check that we fall back to using an counter as index"""
    ds = mock_dataset_nonunique.to_xarray_dataset()
    assert len(ds) == 3
    assert "index" in ds.coords
    assert "x" not in ds.coords
    _assert_xarray_metadata_is_as_expected(ds, mock_dataset_nonunique)

    for array_name in ds.data_vars:
        assert "snapshot" not in ds[array_name].attrs.keys()


def test_export_to_xarray_extra_metadata(mock_dataset) -> None:
    mock_dataset.add_metadata("mytag", "somestring")
    mock_dataset.add_metadata("myothertag", 1)
    ds = mock_dataset.to_xarray_dataset()

    _assert_xarray_metadata_is_as_expected(ds, mock_dataset)

    for array_name in ds.data_vars:
        assert "snapshot" not in ds[array_name].attrs.keys()


def test_export_to_xarray_ds_dict_extra_metadata(mock_dataset) -> None:
    mock_dataset.add_metadata("mytag", "somestring")
    mock_dataset.add_metadata("myothertag", 1)
    da_dict = mock_dataset.to_xarray_dataarray_dict()

    for datarray in da_dict.values():
        _assert_xarray_metadata_is_as_expected(datarray, mock_dataset)


def test_export_to_xarray_extra_metadata_can_be_stored(mock_dataset, tmp_path) -> None:
    nt_metadata = {
        "foo": {
            "bar": {"baz": "test"},
            "spam": [1, 2, 3],
        }
    }
    mock_dataset.add_metadata("foo_metadata", json.dumps(nt_metadata))
    mock_dataset.export(export_type="netcdf", path=str(tmp_path))

    mock_dataset.add_metadata("metadata_added_after_export", 69)

    data_as_xarray = mock_dataset.to_xarray_dataset()

    loaded_data = xr.load_dataset(
        tmp_path
        / f"{qcodes.config.dataset.export_prefix}{mock_dataset.captured_run_id}_{mock_dataset.guid}.nc"
    )

    # check that the metadata in the qcodes dataset is roundtripped to the loaded
    # dataset
    # export info is only set after the export so its not part of
    # the exported metadata so skip it here.
    for key in mock_dataset.metadata.keys():
        if key != "export_info":
            assert mock_dataset.metadata[key] == loaded_data.attrs[key]
    # check that the added metadata roundtrip correctly
    assert loaded_data.attrs["foo_metadata"] == json.dumps(nt_metadata)
    assert loaded_data.attrs["metadata_added_after_export"] == 69
    # check that all attrs roundtrip correctly within the xarray ds
    data_as_xarray.attrs.pop("export_info")
    assert loaded_data.attrs == data_as_xarray.attrs


def test_to_xarray_ds_paramspec_metadata_is_preserved(mock_dataset_label_unit) -> None:
    xr_ds = mock_dataset_label_unit.to_xarray_dataset()
    assert len(xr_ds.dims) == 1
    for param_name in xr_ds.dims:
        assert xr_ds.coords[param_name].attrs == _get_expected_param_spec_attrs(
            mock_dataset_label_unit, param_name
        )
    for param_name in xr_ds.data_vars:
        assert xr_ds.data_vars[param_name].attrs == _get_expected_param_spec_attrs(
            mock_dataset_label_unit, param_name
        )


def test_to_xarray_da_dict_paramspec_metadata_is_preserved(
    mock_dataset_label_unit,
) -> None:
    xr_das = mock_dataset_label_unit.to_xarray_dataarray_dict()

    for outer_param_name, xr_da in xr_das.items():
        for param_name in xr_da.dims:
            assert xr_da.coords[param_name].attrs == _get_expected_param_spec_attrs(
                mock_dataset_label_unit, param_name
            )
        expected_param_spec_attrs = _get_expected_param_spec_attrs(
            mock_dataset_label_unit, outer_param_name
        )
        for spec_name, spec_value in expected_param_spec_attrs.items():
            assert xr_da.attrs[spec_name] == spec_value


def test_export_2d_dataset(
    tmp_path_factory: TempPathFactory, mock_dataset_grid: DataSet
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    path = str(tmp_path)
    mock_dataset_grid.export(export_type="netcdf", path=tmp_path, prefix="qcodes_")

    pdf = mock_dataset_grid.to_pandas_dataframe()
    dims = _calculate_index_shape(pdf.index)
    assert dims == {"x": 10, "y": 5}

    xr_ds = mock_dataset_grid.to_xarray_dataset()
    assert xr_ds["z"].dims == ("x", "y")

    expected_path = (
        f"qcodes_{mock_dataset_grid.captured_run_id}_{mock_dataset_grid.guid}.nc"
    )
    assert os.listdir(path) == [expected_path]
    file_path = os.path.join(path, expected_path)
    ds = load_from_netcdf(file_path)

    xr_ds_reimported = ds.to_xarray_dataset()

    assert xr_ds_reimported["z"].dims == ("x", "y")
    assert xr_ds.identical(xr_ds_reimported)


def test_export_dataset_small_no_delated(
    tmp_path_factory: TempPathFactory, mock_dataset_numpy: DataSet, caplog
) -> None:
    """
    Test that a 'small' dataset does not use the delayed export.
    """
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    with caplog.at_level(logging.INFO):
        mock_dataset_numpy.export(export_type="netcdf", path=tmp_path, prefix="qcodes_")

    assert "Writing netcdf file directly" in caplog.records[0].msg


def test_export_dataset_delayed_off_by_default(
    tmp_path_factory: TempPathFactory, mock_dataset_grid: DataSet, caplog
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    qcodes.config.dataset.export_chunked_threshold = 0
    assert qcodes.config.dataset.export_chunked_export_of_large_files_enabled is False
    with caplog.at_level(logging.INFO):
        mock_dataset_grid.export(export_type="netcdf", path=tmp_path, prefix="qcodes_")

    assert "Writing netcdf file directly." in caplog.records[0].msg


def test_export_dataset_delayed_numeric(
    tmp_path_factory: TempPathFactory, mock_dataset_grid: DataSet, caplog
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    qcodes.config.dataset.export_chunked_threshold = 0
    qcodes.config.dataset.export_chunked_export_of_large_files_enabled = True
    with caplog.at_level(logging.INFO):
        mock_dataset_grid.export(export_type="netcdf", path=tmp_path, prefix="qcodes_")

    assert (
        "Dataset is expected to be larger that threshold. Using distributed export."
        in caplog.records[0].msg
    )
    assert "Writing individual files to temp dir" in caplog.records[1].msg
    assert "Combining temp files into one file" in caplog.records[2].msg
    assert "Writing netcdf file using Dask delayed writer" in caplog.records[3].msg

    loaded_ds = xr.load_dataset(mock_dataset_grid.export_info.export_paths["nc"])
    assert loaded_ds.x.shape == (10,)
    assert_allclose(loaded_ds.x, np.arange(10))
    assert loaded_ds.y.shape == (5,)
    assert_allclose(loaded_ds.y, np.arange(20, 25, 1))

    arrays = []
    for i in range(10):
        arrays.append(np.arange(20 + i, 25 + i))
    expected_z = np.array(arrays)

    assert loaded_ds.z.shape == (10, 5)
    assert_allclose(loaded_ds.z, expected_z)

    _assert_xarray_metadata_is_as_expected(loaded_ds, mock_dataset_grid)


def test_export_dataset_delayed(
    tmp_path_factory: TempPathFactory, mock_dataset_numpy: DataSet, caplog
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    qcodes.config.dataset.export_chunked_threshold = 0
    qcodes.config.dataset.export_chunked_export_of_large_files_enabled = True
    with caplog.at_level(logging.INFO):
        mock_dataset_numpy.export(export_type="netcdf", path=tmp_path, prefix="qcodes_")

    assert (
        "Dataset is expected to be larger that threshold. Using distributed export."
        in caplog.records[0].msg
    )
    assert "Writing individual files to temp dir" in caplog.records[1].msg
    assert "Combining temp files into one file" in caplog.records[2].msg
    assert "Writing netcdf file using Dask delayed writer" in caplog.records[3].msg

    loaded_ds = xr.load_dataset(mock_dataset_numpy.export_info.export_paths["nc"])
    assert loaded_ds.x.shape == (10,)
    assert_allclose(loaded_ds.x, np.arange(10))
    assert loaded_ds.y.shape == (11,)
    assert_allclose(loaded_ds.y, np.arange(10, 21, 1))

    arrays = []
    for i in range(10):
        arrays.append(np.arange(10 + i, 21 + i))
    expected_z = np.array(arrays)

    assert loaded_ds.z.shape == (10, 11)
    assert_allclose(loaded_ds.z, expected_z)

    _assert_xarray_metadata_is_as_expected(loaded_ds, mock_dataset_numpy)


def test_export_dataset_delayed_complex(
    tmp_path_factory: TempPathFactory, mock_dataset_numpy_complex: DataSet, caplog
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    qcodes.config.dataset.export_chunked_threshold = 0
    qcodes.config.dataset.export_chunked_export_of_large_files_enabled = True
    with caplog.at_level(logging.INFO):
        mock_dataset_numpy_complex.export(
            export_type="netcdf", path=tmp_path, prefix="qcodes_"
        )

    assert (
        "Dataset is expected to be larger that threshold. Using distributed export."
        in caplog.records[0].msg
    )
    assert "Writing individual files to temp dir" in caplog.records[1].msg
    assert "Combining temp files into one file" in caplog.records[2].msg
    assert "Writing netcdf file using Dask delayed writer" in caplog.records[3].msg

    loaded_ds = xr.load_dataset(
        mock_dataset_numpy_complex.export_info.export_paths["nc"]
    )
    assert loaded_ds.x.shape == (10,)
    assert_allclose(loaded_ds.x, np.arange(10))
    assert loaded_ds.y.shape == (11,)
    assert_allclose(loaded_ds.y, np.arange(10, 21, 1))

    arrays = []
    for i in range(10):
        arrays.append(1j * np.arange(10, 21) + i)
    expected_z = np.array(arrays)

    assert loaded_ds.z.shape == (10, 11)
    assert_allclose(loaded_ds.z, expected_z)
    _assert_xarray_metadata_is_as_expected(loaded_ds, mock_dataset_numpy_complex)


def test_export_non_grid_dataset_xarray(mock_dataset_non_grid: DataSet) -> None:
    xr_ds = mock_dataset_non_grid.to_xarray_dataset()
    assert xr_ds.sizes == {"multi_index": 50}
    assert len(xr_ds.coords) == 3  # dims + 1 multi index
    assert "x" in xr_ds.coords
    assert len(xr_ds.coords["x"].attrs) == 8
    assert "y" in xr_ds.coords
    assert len(xr_ds.coords["y"].attrs) == 8
    assert "multi_index" in xr_ds.coords
    assert len(xr_ds.coords["multi_index"].attrs) == 0


def test_export_non_grid_in_grid_dataset_xarray(
    mock_dataset_non_grid_in_grid: DataSet,
) -> None:
    xr_ds = mock_dataset_non_grid_in_grid.to_xarray_dataset()
    # this is a dataset where we sweep x from 1 -> 9
    # for each x we measure 50 points as a function of random values of y1 and y2

    assert len(xr_ds.coords) == 4  # dims + 1 multi index
    assert xr_ds.sizes == {"multi_index": 450}
    # ideally we would probably expect this to be {x: 9, multi_index: 50}
    # however at the moment we do not store the "multiindexed" parameters
    # seperately from the "regular" index parameters when there is a multiindex
    # parameter

    assert "x" in xr_ds.coords
    assert len(xr_ds.coords["x"].attrs) == 8
    assert "y1" in xr_ds.coords
    assert len(xr_ds.coords["y1"].attrs) == 8
    assert "y2" in xr_ds.coords
    assert len(xr_ds.coords["y2"].attrs) == 8
    assert "multi_index" in xr_ds.coords
    assert len(xr_ds.coords["multi_index"].attrs) == 0


def test_export_non_grid_dataset(
    tmp_path_factory: TempPathFactory, mock_dataset_non_grid: DataSet
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    path = str(tmp_path)
    mock_dataset_non_grid.export(export_type="netcdf", path=tmp_path, prefix="qcodes_")

    pdf = mock_dataset_non_grid.to_pandas_dataframe()
    dims = _calculate_index_shape(pdf.index)
    assert dims == {"x": 50, "y": 50}

    xr_ds = mock_dataset_non_grid.to_xarray_dataset()
    assert len(xr_ds.coords) == 3
    assert "multi_index" in xr_ds.coords
    assert "x" in xr_ds.coords
    assert "y" in xr_ds.coords
    assert xr_ds.sizes == {"multi_index": 50}

    expected_path = f"qcodes_{mock_dataset_non_grid.captured_run_id}_{mock_dataset_non_grid.guid}.nc"
    assert os.listdir(path) == [expected_path]
    file_path = os.path.join(path, expected_path)
    ds = load_from_netcdf(file_path)

    xr_ds_reimported = ds.to_xarray_dataset()

    assert len(xr_ds_reimported.coords) == 3
    assert "multi_index" in xr_ds_reimported.coords
    assert "x" in xr_ds_reimported.coords
    assert "y" in xr_ds_reimported.coords
    assert xr_ds.identical(xr_ds_reimported)


def test_export_non_grid_in_mem_dataset(
    tmp_path_factory: TempPathFactory, mock_dataset_non_grid_in_mem: DataSetProtocol
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    path = str(tmp_path)
    mock_dataset_non_grid_in_mem.export(
        export_type="netcdf", path=tmp_path, prefix="qcodes_"
    )

    pdf = mock_dataset_non_grid_in_mem.to_pandas_dataframe()
    dims = _calculate_index_shape(pdf.index)
    assert dims == {"x": 50, "y": 50}

    xr_ds = mock_dataset_non_grid_in_mem.to_xarray_dataset()
    assert len(xr_ds.coords) == 3
    assert "multi_index" in xr_ds.coords
    assert "x" in xr_ds.coords
    assert "y" in xr_ds.coords
    assert xr_ds.sizes == {"multi_index": 50}

    expected_path = f"qcodes_{mock_dataset_non_grid_in_mem.captured_run_id}_{mock_dataset_non_grid_in_mem.guid}.nc"
    assert os.listdir(path) == [expected_path]
    file_path = os.path.join(path, expected_path)
    ds = load_from_netcdf(file_path)

    xr_ds_reimported = ds.to_xarray_dataset()

    assert len(xr_ds_reimported.coords) == 3
    assert "multi_index" in xr_ds_reimported.coords
    assert "x" in xr_ds_reimported.coords
    assert "y" in xr_ds_reimported.coords
    assert xr_ds.identical(xr_ds_reimported)

    loaded_by_id_ds = load_by_id(mock_dataset_non_grid_in_mem.run_id)
    loaded_by_id_ds_xr = loaded_by_id_ds.to_xarray_dataset()

    assert len(loaded_by_id_ds_xr.coords) == 3
    assert "multi_index" in loaded_by_id_ds_xr.coords
    assert "x" in loaded_by_id_ds_xr.coords
    assert "y" in loaded_by_id_ds_xr.coords
    assert xr_ds.identical(loaded_by_id_ds_xr)


def test_export_non_grid_in_grid_dataset(
    tmp_path_factory: TempPathFactory, mock_dataset_non_grid_in_grid: DataSet
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    path = str(tmp_path)
    mock_dataset_non_grid_in_grid.export(
        export_type="netcdf", path=tmp_path, prefix="qcodes_"
    )

    pdf = mock_dataset_non_grid_in_grid.to_pandas_dataframe()
    dims = _calculate_index_shape(pdf.index)
    assert dims == {"x": 9, "y1": 450, "y2": 450}

    xr_ds = mock_dataset_non_grid_in_grid.to_xarray_dataset()
    assert len(xr_ds.coords) == 4
    assert "multi_index" in xr_ds.coords
    assert "x" in xr_ds.coords
    assert "y1" in xr_ds.coords
    assert "y2" in xr_ds.coords

    assert xr_ds.sizes == {"multi_index": 450}

    expected_path = f"qcodes_{mock_dataset_non_grid_in_grid.captured_run_id}_{mock_dataset_non_grid_in_grid.guid}.nc"
    assert os.listdir(path) == [expected_path]
    file_path = os.path.join(path, expected_path)
    ds = load_from_netcdf(file_path)

    xr_ds_reimported = ds.to_xarray_dataset()

    assert len(xr_ds_reimported.coords) == 4
    assert "multi_index" in xr_ds_reimported.coords
    assert "x" in xr_ds_reimported.coords
    assert "y1" in xr_ds_reimported.coords
    assert "y2" in xr_ds_reimported.coords
    assert xr_ds.identical(xr_ds_reimported)


def test_export_grid_in_non_grid_dataset(
    tmp_path_factory: TempPathFactory, mock_dataset_grid_in_non_grid: DataSet
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    path = str(tmp_path)
    mock_dataset_grid_in_non_grid.export(
        export_type="netcdf", path=tmp_path, prefix="qcodes_"
    )

    pdf = mock_dataset_grid_in_non_grid.to_pandas_dataframe()
    dims = _calculate_index_shape(pdf.index)
    assert dims == {"x1": 10, "x2": 10, "y": 5}

    xr_ds = mock_dataset_grid_in_non_grid.to_xarray_dataset()
    assert len(xr_ds.coords) == 4
    assert "multi_index" in xr_ds.coords
    assert "x1" in xr_ds.coords
    assert "x2" in xr_ds.coords
    assert "y" in xr_ds.coords

    assert xr_ds.sizes == {"multi_index": 50}

    expected_path = f"qcodes_{mock_dataset_grid_in_non_grid.captured_run_id}_{mock_dataset_grid_in_non_grid.guid}.nc"
    assert os.listdir(path) == [expected_path]
    file_path = os.path.join(path, expected_path)
    ds = load_from_netcdf(file_path)

    xr_ds_reimported = ds.to_xarray_dataset()

    assert len(xr_ds_reimported.coords) == 4
    assert "multi_index" in xr_ds_reimported.coords
    assert "x1" in xr_ds_reimported.coords
    assert "x2" in xr_ds_reimported.coords
    assert "y" in xr_ds_reimported.coords
    assert xr_ds.identical(xr_ds_reimported)


def test_export_non_grid_in_non_grid_dataset(
    tmp_path_factory: TempPathFactory, mock_dataset_non_grid_in_non_grid: DataSet
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    path = str(tmp_path)
    mock_dataset_non_grid_in_non_grid.export(
        export_type="netcdf", path=tmp_path, prefix="qcodes_"
    )

    pdf = mock_dataset_non_grid_in_non_grid.to_pandas_dataframe()
    dims = _calculate_index_shape(pdf.index)
    assert dims == {"x1": 10, "x2": 10, "y1": 50, "y2": 50}

    xr_ds = mock_dataset_non_grid_in_non_grid.to_xarray_dataset()
    assert len(xr_ds.coords) == 5
    assert "multi_index" in xr_ds.coords
    assert "x1" in xr_ds.coords
    assert "x2" in xr_ds.coords
    assert "y1" in xr_ds.coords
    assert "y2" in xr_ds.coords

    assert xr_ds.sizes == {"multi_index": 50}

    expected_path = f"qcodes_{mock_dataset_non_grid_in_non_grid.captured_run_id}_{mock_dataset_non_grid_in_non_grid.guid}.nc"
    assert os.listdir(path) == [expected_path]
    file_path = os.path.join(path, expected_path)
    ds = load_from_netcdf(file_path)

    xr_ds_reimported = ds.to_xarray_dataset()

    assert len(xr_ds_reimported.coords) == 5
    assert "multi_index" in xr_ds_reimported.coords
    assert "x1" in xr_ds_reimported.coords
    assert "x2" in xr_ds_reimported.coords
    assert "y1" in xr_ds_reimported.coords
    assert "y2" in xr_ds_reimported.coords
    assert xr_ds.identical(xr_ds_reimported)


def test_inverted_coords_perserved_on_netcdf_roundtrip(
    tmp_path_factory: TempPathFactory, mock_dataset_inverted_coords
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    path = str(tmp_path)
    mock_dataset_inverted_coords.export(
        export_type="netcdf", path=tmp_path, prefix="qcodes_"
    )

    xr_ds = mock_dataset_inverted_coords.to_xarray_dataset()
    assert xr_ds["z1"].dims == ("x", "y")
    assert xr_ds["z2"].dims == ("y", "x")

    expected_path = f"qcodes_{mock_dataset_inverted_coords.captured_run_id}_{mock_dataset_inverted_coords.guid}.nc"
    assert os.listdir(path) == [expected_path]
    file_path = os.path.join(path, expected_path)
    ds = load_from_netcdf(file_path)

    xr_ds_reimported = ds.to_xarray_dataset()

    assert xr_ds_reimported["z1"].dims == ("x", "y")
    assert xr_ds_reimported["z2"].dims == ("y", "x")
    assert xr_ds.identical(xr_ds_reimported)


def _get_expected_param_spec_attrs(dataset, dim):
    expected_attrs = dict(dataset.paramspecs[str(dim)]._to_dict())
    expected_attrs["units"] = expected_attrs["unit"]
    expected_attrs["long_name"] = expected_attrs["label"]
    assert len(expected_attrs.keys()) == 8
    return expected_attrs


def _assert_xarray_metadata_is_as_expected(xarray_ds, qc_dataset):
    assert xarray_ds.ds_name == qc_dataset.name
    assert xarray_ds.sample_name == qc_dataset.sample_name
    assert xarray_ds.exp_name == qc_dataset.exp_name
    assert (
        xarray_ds.snapshot == qc_dataset.snapshot_raw
        if qc_dataset.snapshot_raw is not None
        else "null"
    )
    assert xarray_ds.guid == qc_dataset.guid
    assert xarray_ds.run_timestamp == qc_dataset.run_timestamp()
    assert xarray_ds.completed_timestamp == qc_dataset.completed_timestamp()
    assert xarray_ds.captured_run_id == qc_dataset.captured_run_id
    assert xarray_ds.captured_counter == qc_dataset.captured_counter
    assert xarray_ds.run_id == qc_dataset.run_id
    assert xarray_ds.run_description == serial.to_json_for_storage(
        qc_dataset.description
    )
    assert xarray_ds.parent_dataset_links == links_to_str(
        qc_dataset.parent_dataset_links
    )


def test_multi_index_options_grid(mock_dataset_grid) -> None:
    assert mock_dataset_grid.description.shapes is None

    xds = mock_dataset_grid.to_xarray_dataset()
    assert xds.sizes == {"x": 10, "y": 5}

    xds_never = mock_dataset_grid.to_xarray_dataset(use_multi_index="never")
    assert xds_never.sizes == {"x": 10, "y": 5}

    xds_auto = mock_dataset_grid.to_xarray_dataset(use_multi_index="auto")
    assert xds_auto.sizes == {"x": 10, "y": 5}

    xds_always = mock_dataset_grid.to_xarray_dataset(use_multi_index="always")
    assert xds_always.sizes == {"multi_index": 50}


def test_multi_index_options_grid_with_shape(mock_dataset_grid_with_shapes) -> None:
    assert mock_dataset_grid_with_shapes.description.shapes == {"z": (10, 5)}

    xds = mock_dataset_grid_with_shapes.to_xarray_dataset()
    assert xds.sizes == {"x": 10, "y": 5}

    xds_never = mock_dataset_grid_with_shapes.to_xarray_dataset(use_multi_index="never")
    assert xds_never.sizes == {"x": 10, "y": 5}

    xds_auto = mock_dataset_grid_with_shapes.to_xarray_dataset(use_multi_index="auto")
    assert xds_auto.sizes == {"x": 10, "y": 5}

    xds_always = mock_dataset_grid_with_shapes.to_xarray_dataset(
        use_multi_index="always"
    )
    assert xds_always.sizes == {"multi_index": 50}


def test_multi_index_options_incomplete_grid(mock_dataset_grid_incomplete) -> None:
    assert mock_dataset_grid_incomplete.description.shapes is None

    xds = mock_dataset_grid_incomplete.to_xarray_dataset()
    assert xds.sizes == {"multi_index": 39}

    xds_never = mock_dataset_grid_incomplete.to_xarray_dataset(use_multi_index="never")
    assert xds_never.sizes == {"x": 8, "y": 5}

    xds_auto = mock_dataset_grid_incomplete.to_xarray_dataset(use_multi_index="auto")
    assert xds_auto.sizes == {"multi_index": 39}

    xds_always = mock_dataset_grid_incomplete.to_xarray_dataset(
        use_multi_index="always"
    )
    assert xds_always.sizes == {"multi_index": 39}


def test_multi_index_options_incomplete_grid_with_shapes(
    mock_dataset_grid_incomplete_with_shapes,
) -> None:
    assert mock_dataset_grid_incomplete_with_shapes.description.shapes == {"z": (10, 5)}

    xds = mock_dataset_grid_incomplete_with_shapes.to_xarray_dataset()
    assert xds.sizes == {"x": 8, "y": 5}

    xds_never = mock_dataset_grid_incomplete_with_shapes.to_xarray_dataset(
        use_multi_index="never"
    )
    assert xds_never.sizes == {"x": 8, "y": 5}

    xds_auto = mock_dataset_grid_incomplete_with_shapes.to_xarray_dataset(
        use_multi_index="auto"
    )
    assert xds_auto.sizes == {"x": 8, "y": 5}

    xds_always = mock_dataset_grid_incomplete_with_shapes.to_xarray_dataset(
        use_multi_index="always"
    )
    assert xds_always.sizes == {"multi_index": 39}


def test_multi_index_options_non_grid(mock_dataset_non_grid) -> None:
    assert mock_dataset_non_grid.description.shapes is None

    xds = mock_dataset_non_grid.to_xarray_dataset()
    assert xds.sizes == {"multi_index": 50}

    xds_never = mock_dataset_non_grid.to_xarray_dataset(use_multi_index="never")
    assert xds_never.sizes == {"x": 50, "y": 50}

    xds_auto = mock_dataset_non_grid.to_xarray_dataset(use_multi_index="auto")
    assert xds_auto.sizes == {"multi_index": 50}

    xds_always = mock_dataset_non_grid.to_xarray_dataset(use_multi_index="always")
    assert xds_always.sizes == {"multi_index": 50}


def test_multi_index_wrong_option(mock_dataset_non_grid) -> None:
    with pytest.raises(ValueError, match="Invalid value for use_multi_index"):
        mock_dataset_non_grid.to_xarray_dataset(use_multi_index=True)

    with pytest.raises(ValueError, match="Invalid value for use_multi_index"):
        mock_dataset_non_grid.to_xarray_dataset(use_multi_index=False)

    with pytest.raises(ValueError, match="Invalid value for use_multi_index"):
        mock_dataset_non_grid.to_xarray_dataset(use_multi_index="perhaps")


def test_geneate_pandas_index():
    indexes = {
        "z": np.array([[7, 8, 9], [10, 11, 12]]),
        "x": np.array([[1, 2, 3], [1, 2, 3]]),
        "y": np.array([[5, 5, 5], [6, 6, 6]]),
    }
    pdi = _generate_pandas_index(indexes)
    assert isinstance(pdi, pd.MultiIndex)
    assert len(pdi) == 6

    indexes = {
        "z": np.array([[7, 8, 9], [10, 11, 12]]),
        "x": np.array([["a", "b", "c"], ["a", "b", "c"]]),
        "y": np.array([[5, 5, 5], [6, 6, 6]]),
    }
    pdi = _generate_pandas_index(indexes)
    assert isinstance(pdi, pd.MultiIndex)
    assert len(pdi) == 6

    indexes = {
        "z": np.array([[7, 8, 9], [10, 11, 12]]),
        "x": np.array([["a", "b", "c"], ["a", "b", "c"]], dtype=np.object_),
        "y": np.array([[5, 5, 5], [6, 6, 6]]),
    }
    pdi = _generate_pandas_index(indexes)
    assert isinstance(pdi, pd.MultiIndex)
    assert len(pdi) == 6

    indexes = {
        "z": np.array([[7], [8, 9]], dtype=np.object_),
        "x": np.array([["a"], ["a", "b"]], dtype=np.object_),
        "y": np.array([[5], [6, 6]], dtype=np.object_),
    }
    pdi = _generate_pandas_index(indexes)
    assert isinstance(pdi, pd.MultiIndex)
    assert len(pdi) == 3


@given(
    function_name=hst.sampled_from(
        [
            "to_xarray_dataarray_dict",
            "to_pandas_dataframe",
            "to_pandas_dataframe_dict",
            "get_parameter_data",
        ]
    )
)
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,), deadline=None)
def test_export_lazy_load(
    tmp_path_factory: TempPathFactory, mock_dataset_grid: DataSet, function_name: str
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    path = str(tmp_path)
    mock_dataset_grid.export(export_type="netcdf", path=tmp_path, prefix="qcodes_")

    xr_ds = mock_dataset_grid.to_xarray_dataset()
    assert xr_ds["z"].dims == ("x", "y")

    expected_path = (
        f"qcodes_{mock_dataset_grid.captured_run_id}_{mock_dataset_grid.guid}.nc"
    )
    assert os.listdir(path) == [expected_path]
    file_path = os.path.join(path, expected_path)
    ds = load_from_netcdf(file_path)

    # loading the dataset should not load the actual data into cache
    assert ds.cache._data == {}
    # loading directly into xarray should not round
    # trip to qcodes format and  therefor not fill the cache
    xr_ds_reimported = ds.to_xarray_dataset()
    assert ds.cache._data == {}

    assert xr_ds_reimported["z"].dims == ("x", "y")
    assert xr_ds.identical(xr_ds_reimported)

    # but loading with any of these functions
    # will currently fill the cache
    getattr(ds, function_name)()

    assert ds.cache._data != {}


@given(
    function_name=hst.sampled_from(
        [
            "to_xarray_dataarray_dict",
            "to_pandas_dataframe",
            "to_pandas_dataframe_dict",
            "get_parameter_data",
        ]
    )
)
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,), deadline=None)
def test_export_lazy_load_in_mem_dataset(
    tmp_path_factory: TempPathFactory,
    mock_dataset_in_mem_grid: DataSet,
    function_name: str,
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    path = str(tmp_path)
    mock_dataset_in_mem_grid.export(
        export_type="netcdf", path=tmp_path, prefix="qcodes_"
    )

    xr_ds = mock_dataset_in_mem_grid.to_xarray_dataset()
    assert xr_ds["z"].dims == ("x", "y")

    expected_path = f"qcodes_{mock_dataset_in_mem_grid.captured_run_id}_{mock_dataset_in_mem_grid.guid}.nc"
    assert os.listdir(path) == [expected_path]
    file_path = os.path.join(path, expected_path)
    ds = load_from_netcdf(file_path)

    # loading the dataset should not load the actual data into cache
    assert ds.cache._data == {}
    # loading directly into xarray should not round
    # trip to qcodes format and  therefor not fill the cache
    xr_ds_reimported = ds.to_xarray_dataset()
    assert ds.cache._data == {}

    assert xr_ds_reimported["z"].dims == ("x", "y")
    assert xr_ds.identical(xr_ds_reimported)

    # but loading with any of these functions
    # will currently fill the cache
    getattr(ds, function_name)()

    assert ds.cache._data != {}

    dataset_loaded_by_guid = load_by_guid(mock_dataset_in_mem_grid.guid)

    # loading the dataset should not load the actual data into cache
    assert dataset_loaded_by_guid.cache._data == {}
    # loading directly into xarray should not round
    # trip to qcodes format and  therefor not fill the cache
    xr_ds_reimported = dataset_loaded_by_guid.to_xarray_dataset()
    assert dataset_loaded_by_guid.cache._data == {}

    assert xr_ds_reimported["z"].dims == ("x", "y")
    assert xr_ds.identical(xr_ds_reimported)

    # but loading with any of these functions
    # will currently fill the cache
    getattr(dataset_loaded_by_guid, function_name)()

    assert dataset_loaded_by_guid.cache._data != {}
