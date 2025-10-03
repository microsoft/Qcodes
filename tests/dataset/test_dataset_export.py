from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as hst
from numpy.testing import assert_allclose, assert_array_equal
from pytest import LogCaptureFixture, TempPathFactory

import qcodes
from qcodes.dataset import (
    DataSetProtocol,
    DataSetType,
    LinSweep,
    Measurement,
    dond,
    get_data_export_path,
    load_by_guid,
    load_by_id,
    load_from_netcdf,
    new_data_set,
)
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.data_set_in_memory import DataSetInMem
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.versioning import serialization as serial
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.export_config import DataExportType
from qcodes.dataset.exporters.export_to_pandas import _generate_pandas_index
from qcodes.dataset.exporters.export_to_xarray import _calculate_index_shape
from qcodes.dataset.linked_datasets.links import links_to_str
from qcodes.parameters import ManualParameter, Parameter, ParamSpecBase
from qcodes.utils.deprecate import QCoDeSDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pytest_mock import MockerFixture

    from qcodes.dataset.data_set import DataSet
    from qcodes.dataset.experiment_container import Experiment


@pytest.fixture(name="mock_empty_dataset")
def _make_mock_empty_dataset(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_nonunique_index(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_label_unit(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_complex(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_grid(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_in_mem_grid(experiment: Experiment) -> DataSetProtocol:
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
def _make_mock_dataset_grid_with_shapes(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_grid_incomplete(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_grid_incomplete_with_shapes(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_numpy(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_numpy_complex(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_non_grid(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_non_grid_in_mem(experiment: Experiment) -> DataSetProtocol:
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
def _make_mock_dataset_non_grid_in_grid(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_grid_in_non_grid(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_non_grid_in_non_grid(experiment: Experiment) -> DataSet:
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
def _make_mock_dataset_inverted_coords(experiment: Experiment) -> DataSet:
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
def test_write_data_to_text_file_save(tmp_path_factory: TempPathFactory) -> None:
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
    tmp_path_factory: TempPathFactory, mock_dataset: DataSet
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
    tmp_path_factory: TempPathFactory, mock_dataset: DataSet
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
def test_write_data_to_text_file_length_exception(tmp_path: Path) -> None:
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


def test_write_data_to_text_file_name_exception(
    tmp_path: Path, mock_dataset: DataSet
) -> None:
    temp_dir = str(tmp_path)
    with pytest.raises(Exception, match="desired file name"):
        mock_dataset.write_data_to_text_file(
            path=temp_dir, single_file=True, single_file_name=None
        )


def test_export_csv(
    tmp_path_factory: TempPathFactory, mock_dataset: DataSet, caplog: LogCaptureFixture
) -> None:
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
    tmp_path_factory: TempPathFactory, mock_dataset: DataSet, caplog: LogCaptureFixture
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
    tmp_path_factory: TempPathFactory, mock_dataset: DataSet
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


def test_export_netcdf_csv(
    tmp_path_factory: TempPathFactory, mock_dataset: DataSet
) -> None:
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


def test_export_netcdf_complex_data(
    tmp_path_factory: TempPathFactory, mock_dataset_complex: DataSet
) -> None:
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
    tmp_path_factory: TempPathFactory, mock_dataset: DataSet
) -> None:
    with pytest.raises(ValueError, match="No data export type specified"):
        mock_dataset.export()

    with pytest.raises(ValueError, match=re.escape("Export type foo is unknown.")):
        mock_dataset.export(export_type="foo")


def test_export_from_config(
    tmp_path_factory: TempPathFactory, mock_dataset: DataSet, mocker: MockerFixture
) -> None:
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
    tmp_path_factory: TempPathFactory, mock_dataset: DataSet, mocker: MockerFixture
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


def test_same_setpoint_warning_for_df_two_params_with_different_setpoints(
    different_setpoint_dataset: DataSetProtocol,
) -> None:
    warning_message = (
        "Independent parameter setpoints are not equal. "
        "Check concatenated output carefully."
    )

    with pytest.warns(UserWarning, match=warning_message):
        df = different_setpoint_dataset.to_pandas_dataframe()
    this_2_7_df = df.loc[df["this_2_7"].notna()]["this_2_7"]
    # Verify that the index for the filtered Series has the expected 2x7 shape
    assert isinstance(this_2_7_df.index, pd.MultiIndex)

    # Names of setpoint coords as registered by the MultiParameter
    # because this dataframe has indexes from two different set of
    # qcodes setpoints the names are just level_0, level_1
    sp_21 = "level_0"
    sp_22 = "level_1"

    # Expected coordinate vectors
    exp_sp_21 = np.linspace(5, 9, 2)
    exp_sp_22 = np.linspace(9, 11, 7)

    # Expected shape for the MultiIndex
    dims = _calculate_index_shape(this_2_7_df.index)
    assert dims == {sp_21: 2, sp_22: 7}

    # Basic sanity checks on index names and size
    assert list(this_2_7_df.index.names) == [None, None]
    assert len(this_2_7_df) == 2 * 7

    # Check that each index level matches expected coordinate values
    mi_clean_2_7 = this_2_7_df.index.remove_unused_levels()
    np.testing.assert_allclose(mi_clean_2_7.levels[0].values, exp_sp_21)
    np.testing.assert_allclose(mi_clean_2_7.levels[1].values, exp_sp_22)

    # Repeat for the other parameter (this_5_3)
    this_5_3_df = df.loc[df["this_5_3"].notna()]["this_5_3"]
    assert isinstance(this_5_3_df.index, pd.MultiIndex)

    sp_11 = "level_0"
    sp_12 = "level_1"

    exp_sp_11 = np.linspace(5, 9, 5)
    exp_sp_12 = np.linspace(9, 11, 3)

    dims = _calculate_index_shape(this_5_3_df.index)
    assert dims == {sp_11: 5, sp_12: 3}

    assert list(this_5_3_df.index.names) == [None, None]
    assert len(this_5_3_df) == 5 * 3

    mi_clean_5_3 = this_5_3_df.index.remove_unused_levels()
    np.testing.assert_allclose(mi_clean_5_3.levels[0].values, exp_sp_11)
    np.testing.assert_allclose(mi_clean_5_3.levels[1].values, exp_sp_12)


def test_same_setpoint_warning_for_df_two_params_partial_overlapping_setpoints(
    two_params_partial_2d_dataset: DataSetProtocol,
) -> None:
    warning_message = (
        "Independent parameter setpoints are not equal. "
        "Check concatenated output carefully."
    )

    # Expect a warning since independent parameter setpoints differ
    with pytest.warns(UserWarning, match=warning_message):
        df = two_params_partial_2d_dataset.to_pandas_dataframe()

    # Expect two measured columns (m1 and m2)
    assert "m1" in df.columns
    assert "m2" in df.columns

    # Dataframe should use a MultiIndex named by the two setpoints
    assert isinstance(df.index, pd.MultiIndex)
    assert list(df.index.names) == ["x", "y"]

    # m1 should cover full 5x4 grid
    m1_series = df.loc[df["m1"].notna(), "m1"]
    assert isinstance(m1_series.index, pd.MultiIndex)
    dims_m1 = _calculate_index_shape(m1_series.index)
    assert dims_m1 == {"x": 5, "y": 4}
    assert len(m1_series) == 5 * 4

    # m2 should be partial (5x2 points not NaN)
    m2_series = df.loc[df["m2"].notna(), "m2"]
    assert isinstance(m2_series.index, pd.MultiIndex)
    dims_m2 = _calculate_index_shape(m2_series.index)
    assert dims_m2 == {"x": 5, "y": 2}
    assert len(m2_series) == 5 * 2


def test_partally_overlapping_setpoint_xarray_export(
    different_setpoint_dataset: DataSetProtocol,
) -> None:
    """
    Test that a dataset with two MultiParameters with different
    setpoints can be exported to xarray.Dataset with the correct
    coordinates and shapes."""
    xrds = different_setpoint_dataset.to_xarray_dataset()

    # Expect two data variables from Multi2DSetPointParam2Sizes
    assert "this_5_3" in xrds.data_vars
    assert "this_2_7" in xrds.data_vars

    # Names of setpoint coords as registered by the MultiParameter
    sp_11 = "multi_2d_setpoint_param_this_setpoint_1"
    sp_12 = "multi_2d_setpoint_param_that_setpoint_1"
    sp_21 = "multi_2d_setpoint_param_this_setpoint_2"
    sp_22 = "multi_2d_setpoint_param_that_setpoint_2"

    # Coordinate vectors expected from the mock parameter definition
    exp_sp_11 = np.linspace(5, 9, 5)
    exp_sp_12 = np.linspace(9, 11, 3)
    exp_sp_21 = np.linspace(5, 9, 2)
    exp_sp_22 = np.linspace(9, 11, 7)

    # Check coordinates exist and match expected values
    for name, exp_vals in (
        (sp_11, exp_sp_11),
        (sp_12, exp_sp_12),
        (sp_21, exp_sp_21),
        (sp_22, exp_sp_22),
    ):
        assert name in xrds.coords
        np.testing.assert_allclose(xrds.coords[name].values, exp_vals)

    # this_5_3 should be on grid (sp_11, sp_12) with shape (5, 3)
    assert xrds["this_5_3"].dims == (sp_11, sp_12)
    assert xrds["this_5_3"].shape == (5, 3)

    # this_2_7 should be on grid (sp_21, sp_22) with shape (2, 7)
    assert xrds["this_2_7"].dims == (sp_21, sp_22)
    assert xrds["this_2_7"].shape == (2, 7)

    # Also verify cache path produces consistent coords
    xrds_cache = different_setpoint_dataset.cache.to_xarray_dataset()
    for name, exp_vals in (
        (sp_11, exp_sp_11),
        (sp_12, exp_sp_12),
        (sp_21, exp_sp_21),
        (sp_22, exp_sp_22),
    ):
        assert name in xrds_cache.coords
        np.testing.assert_allclose(xrds_cache.coords[name].values, exp_vals)


def test_partally_overlapping_setpoint_xarray_export_two_params_partial(
    two_params_partial_2d_dataset: DataSetProtocol,
) -> None:
    """
    Similar to test_partally_overlapping_setpoint_xarray_export but using the
    two_params_partial_2d_dataset fixture. Verify that both data variables are
    present, their dims/coords are consistent. This means that for one parameter
    missing values are filled with NaNs.
    """
    xrds = two_params_partial_2d_dataset.to_xarray_dataset()

    # Expect exactly two data variables
    assert len(xrds.data_vars) == 2

    expected_size = (5, 4)

    # Each variable should be 2D and have matching coords
    for _, da in xrds.data_vars.items():
        assert len(da.dims) == 2
        for dim, size in zip(da.dims, expected_size):
            assert dim in xrds.coords
            assert len(xrds.coords[dim]) == da.sizes[dim]
            assert da.sizes[dim] == size

    filtered_data = xrds["m2"].dropna(dim="y", how="all").dropna(dim="x", how="all")

    assert filtered_data.shape == (5, 2)


def test_export_to_xarray_dataset_empty_ds(mock_empty_dataset: DataSet) -> None:
    ds = mock_empty_dataset.to_xarray_dataset()
    assert len(ds) == 2
    assert len(ds.coords) == 1
    assert "x" in ds.coords
    _assert_xarray_metadata_is_as_expected(ds, mock_empty_dataset)


def test_export_to_xarray_dataarray_empty_ds(mock_empty_dataset: DataSet) -> None:
    with pytest.warns(QCoDeSDeprecationWarning, match="to_xarray_dataarray_dict"):
        dad = mock_empty_dataset.to_xarray_dataarray_dict()  # pyright: ignore[reportDeprecated]
    assert len(dad) == 2
    assert len(dad["y"].coords) == 1
    assert "x" in dad["y"].coords
    assert len(dad["z"].coords) == 1
    assert "x" in dad["z"].coords


def test_export_to_xarray_dataset_dict_empty_ds(mock_empty_dataset: DataSet) -> None:
    dad = mock_empty_dataset.to_xarray_dataset_dict()
    assert len(dad) == 2
    assert len(dad["y"].coords) == 1
    assert "x" in dad["y"].coords
    assert len(dad["z"].coords) == 1
    assert "x" in dad["z"].coords


def test_export_to_xarray(mock_dataset: DataSet) -> None:
    ds = mock_dataset.to_xarray_dataset()
    assert len(ds) == 2
    assert "index" not in ds.coords
    assert "x" in ds.coords
    _assert_xarray_metadata_is_as_expected(ds, mock_dataset)


def test_export_to_xarray_non_unique_dependent_parameter(
    mock_dataset_nonunique: DataSet,
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


def test_export_to_xarray_extra_metadata(mock_dataset: DataSet) -> None:
    mock_dataset.add_metadata("mytag", "somestring")
    mock_dataset.add_metadata("myothertag", 1)
    ds = mock_dataset.to_xarray_dataset()

    _assert_xarray_metadata_is_as_expected(ds, mock_dataset)

    for array_name in ds.data_vars:
        assert "snapshot" not in ds[array_name].attrs.keys()


def test_export_to_xarray_da_dict_extra_metadata(mock_dataset: DataSet) -> None:
    mock_dataset.add_metadata("mytag", "somestring")
    mock_dataset.add_metadata("myothertag", 1)
    with pytest.warns(QCoDeSDeprecationWarning, match="to_xarray_dataarray_dict"):
        da_dict = mock_dataset.to_xarray_dataarray_dict()  # pyright: ignore[reportDeprecated]

    for datarray in da_dict.values():
        _assert_xarray_metadata_is_as_expected(datarray, mock_dataset)


def test_export_to_xarray_ds_dict_extra_metadata(mock_dataset: DataSet) -> None:
    mock_dataset.add_metadata("mytag", "somestring")
    mock_dataset.add_metadata("myothertag", 1)
    da_dict = mock_dataset.to_xarray_dataset_dict()

    for datarray in da_dict.values():
        _assert_xarray_metadata_is_as_expected(datarray, mock_dataset)


def test_export_to_xarray_extra_metadata_can_be_stored(
    mock_dataset: DataSet, tmp_path: Path
) -> None:
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


def test_to_xarray_ds_paramspec_metadata_is_preserved(
    mock_dataset_label_unit: DataSet,
) -> None:
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
    mock_dataset_label_unit: DataSet,
) -> None:
    with pytest.warns(QCoDeSDeprecationWarning, match="to_xarray_dataarray_dict"):
        xr_das = mock_dataset_label_unit.to_xarray_dataarray_dict()  # pyright: ignore[reportDeprecated]

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


def test_to_xarray_ds_dict_paramspec_metadata_is_preserved(
    mock_dataset_label_unit: DataSet,
) -> None:
    xr_das = mock_dataset_label_unit.to_xarray_dataset_dict()

    for outer_param_name, xr_da in xr_das.items():
        for param_name in xr_da.dims:
            assert xr_da.coords[param_name].attrs == _get_expected_param_spec_attrs(
                mock_dataset_label_unit, param_name
            )
        expected_param_spec_attrs = _get_expected_param_spec_attrs(
            mock_dataset_label_unit, outer_param_name
        )
        for spec_name, spec_value in expected_param_spec_attrs.items():
            assert xr_da[outer_param_name].attrs[spec_name] == spec_value


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
    tmp_path_factory: TempPathFactory,
    mock_dataset_numpy: DataSet,
    caplog: LogCaptureFixture,
) -> None:
    """
    Test that a 'small' dataset does not use the delayed export.
    """
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    with caplog.at_level(logging.INFO):
        mock_dataset_numpy.export(export_type="netcdf", path=tmp_path, prefix="qcodes_")

    assert "Writing netcdf file directly" in caplog.records[0].msg


def test_export_dataset_delayed_off_by_default(
    tmp_path_factory: TempPathFactory,
    mock_dataset_grid: DataSet,
    caplog: LogCaptureFixture,
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    qcodes.config.dataset.export_chunked_threshold = 0
    assert qcodes.config.dataset.export_chunked_export_of_large_files_enabled is False
    with caplog.at_level(logging.INFO):
        mock_dataset_grid.export(export_type="netcdf", path=tmp_path, prefix="qcodes_")

    assert "Writing netcdf file directly." in caplog.records[0].msg


def test_export_dataset_delayed_numeric(
    tmp_path_factory: TempPathFactory,
    mock_dataset_grid: DataSet,
    caplog: LogCaptureFixture,
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
    for i in range(2, 52):
        assert "Exporting z to xarray via pandas index" in caplog.records[i].message
    assert "Combining temp files into one file" in caplog.records[52].msg
    assert "Writing netcdf file using Dask delayed writer" in caplog.records[53].msg

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
    tmp_path_factory: TempPathFactory,
    mock_dataset_numpy: DataSet,
    caplog: LogCaptureFixture,
) -> None:
    tmp_path = tmp_path_factory.mktemp("export_netcdf")
    qcodes.config.dataset.export_chunked_threshold = 0
    qcodes.config.dataset.export_chunked_export_of_large_files_enabled = True
    with caplog.at_level(logging.INFO):
        mock_dataset_numpy.export(export_type="netcdf", path=tmp_path, prefix="qcodes_")

    assert len(caplog.records) == 16
    assert (
        "Dataset is expected to be larger that threshold. Using distributed export."
        in caplog.records[0].msg
    )
    assert "Writing individual files to temp dir" in caplog.records[1].msg
    for i in range(2, 12):
        assert "Exporting z to xarray via pandas index" in caplog.records[i].message
    assert "Combining temp files into one file" in caplog.records[12].msg
    assert "Writing netcdf file using Dask delayed writer" in caplog.records[13].msg

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
    tmp_path_factory: TempPathFactory,
    mock_dataset_numpy_complex: DataSet,
    caplog: LogCaptureFixture,
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
    for i in range(2, 12):
        assert "Exporting z to xarray via pandas index" in caplog.records[i].message
    assert "Combining temp files into one file" in caplog.records[12].msg
    assert "Writing netcdf file using Dask delayed writer" in caplog.records[13].msg

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
    tmp_path_factory: TempPathFactory, mock_dataset_inverted_coords: DataSet
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


def _get_expected_param_spec_attrs(dataset: DataSet, dim: Hashable) -> dict[str, Any]:
    expected_attrs = dict(dataset.paramspecs[str(dim)]._to_dict())
    expected_attrs["units"] = expected_attrs["unit"]
    expected_attrs["long_name"] = expected_attrs["label"]
    assert len(expected_attrs.keys()) == 8
    return expected_attrs


def _assert_xarray_metadata_is_as_expected(
    xarray_ds: xr.Dataset | xr.DataArray, qc_dataset: DataSet
) -> None:
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


def test_multi_index_options_grid(mock_dataset_grid: DataSet) -> None:
    assert mock_dataset_grid.description.shapes is None

    xds = mock_dataset_grid.to_xarray_dataset()
    assert xds.sizes == {"x": 10, "y": 5}

    xds_never = mock_dataset_grid.to_xarray_dataset(use_multi_index="never")
    assert xds_never.sizes == {"x": 10, "y": 5}

    xds_auto = mock_dataset_grid.to_xarray_dataset(use_multi_index="auto")
    assert xds_auto.sizes == {"x": 10, "y": 5}

    xds_always = mock_dataset_grid.to_xarray_dataset(use_multi_index="always")
    assert xds_always.sizes == {"multi_index": 50}


def test_multi_index_options_grid_with_shape(
    mock_dataset_grid_with_shapes: DataSet,
) -> None:
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


def test_multi_index_options_incomplete_grid(
    mock_dataset_grid_incomplete: DataSet,
) -> None:
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
    mock_dataset_grid_incomplete_with_shapes: DataSet,
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


def test_multi_index_options_non_grid(mock_dataset_non_grid: DataSet) -> None:
    assert mock_dataset_non_grid.description.shapes is None

    xds = mock_dataset_non_grid.to_xarray_dataset()
    assert xds.sizes == {"multi_index": 50}

    xds_never = mock_dataset_non_grid.to_xarray_dataset(use_multi_index="never")
    assert xds_never.sizes == {"x": 50, "y": 50}

    xds_auto = mock_dataset_non_grid.to_xarray_dataset(use_multi_index="auto")
    assert xds_auto.sizes == {"multi_index": 50}

    xds_always = mock_dataset_non_grid.to_xarray_dataset(use_multi_index="always")
    assert xds_always.sizes == {"multi_index": 50}


def test_multi_index_wrong_option(mock_dataset_non_grid: DataSet) -> None:
    with pytest.raises(ValueError, match="Invalid value for use_multi_index"):
        mock_dataset_non_grid.to_xarray_dataset(use_multi_index=True)  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError, match="Invalid value for use_multi_index"):
        mock_dataset_non_grid.to_xarray_dataset(use_multi_index=False)  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError, match="Invalid value for use_multi_index"):
        mock_dataset_non_grid.to_xarray_dataset(use_multi_index="perhaps")  # pyright: ignore[reportArgumentType]


def test_geneate_pandas_index() -> None:
    x = ParamSpecBase("x", "numeric")
    y = ParamSpecBase("y", "numeric")
    z = ParamSpecBase("z", "numeric")

    interdeps = InterDependencies_(dependencies={z: (x, y)})

    data = {
        "z": np.array([[7, 8, 9], [10, 11, 12]]),
        "x": np.array([[1, 2, 3], [1, 2, 3]]),
        "y": np.array([[5, 5, 5], [6, 6, 6]]),
    }
    pdi = _generate_pandas_index(data, interdeps, "z")
    assert isinstance(pdi, pd.MultiIndex)
    assert len(pdi) == 6

    data = {
        "z": np.array([[7, 8, 9], [10, 11, 12]]),
        "x": np.array([["a", "b", "c"], ["a", "b", "c"]]),
        "y": np.array([[5, 5, 5], [6, 6, 6]]),
    }
    pdi = _generate_pandas_index(data, interdeps, "z")
    assert isinstance(pdi, pd.MultiIndex)
    assert len(pdi) == 6

    data = {
        "z": np.array([[7, 8, 9], [10, 11, 12]]),
        "x": np.array([["a", "b", "c"], ["a", "b", "c"]], dtype=np.object_),
        "y": np.array([[5, 5, 5], [6, 6, 6]]),
    }
    pdi = _generate_pandas_index(data, interdeps, "z")
    assert isinstance(pdi, pd.MultiIndex)
    assert len(pdi) == 6

    data = {
        "z": np.array([[7], [8, 9]], dtype=np.object_),
        "x": np.array([["a"], ["a", "b"]], dtype=np.object_),
        "y": np.array([[5], [6, 6]], dtype=np.object_),
    }
    pdi = _generate_pandas_index(data, interdeps, "z")
    assert isinstance(pdi, pd.MultiIndex)
    assert len(pdi) == 3


@given(
    function_name=hst.sampled_from(
        [
            "to_xarray_dataset_dict",
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
            "to_xarray_dataset_dict",
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


@given(data=hst.data())
@settings(
    max_examples=10,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
    deadline=None,
)
def test_dond_hypothesis_nd_grid(
    data: hst.DataObject, experiment: Experiment, caplog: LogCaptureFixture
) -> None:
    """
    Randomized ND sweep using dond:
    - Draw N in [1, 4]
    - For each dimension i, draw number of points n_i in [1, 5]
    - Sweep each ManualParameter over a linspace of length n_i
    - Measure a deterministic function of the setpoints
    - Assert xarray dims, coords, and data match expectation
    """
    n_dims = data.draw(hst.integers(min_value=1, max_value=4), label="n_dims")
    points_per_dim = [
        data.draw(hst.integers(min_value=1, max_value=5), label=f"n_points_dim_{i}")
        for i in range(n_dims)
    ]

    # Create sweep parameters and corresponding value arrays
    sweeps: list[LinSweep] = []
    sweep_params: list[Parameter] = []
    sweep_values: list[np.ndarray] = []
    for i, npts in enumerate(points_per_dim):
        p = ManualParameter(name=f"x{i}")
        sweeps.append(LinSweep(p, 0.0, float(npts - 1), npts))
        vals = np.linspace(0.0, float(npts - 1), npts)
        sweep_params.append(p)
        sweep_values.append(vals)

    # Deterministic measurement as weighted sum of current setpoints
    weights = [(i + 1) for i in range(n_dims)]

    meas_param = Parameter(
        name="signal",
        get_cmd=lambda: float(
            sum(weights[i] * float(sweep_params[i].get()) for i in range(n_dims))
        ),
        set_cmd=None,
    )

    # Build dond and run
    result = dond(
        *sweeps,
        meas_param,
        do_plot=False,
        show_progress=False,
        exp=experiment,
        squeeze=False,
    )

    ds = result[0][0]

    caplog.clear()
    with caplog.at_level(logging.INFO):
        xr_ds = ds.to_xarray_dataset()

    any(
        "Exporting signal to xarray using direct method" in record.message
        for record in caplog.records
    )
    # Expected sizes per coordinate
    expected_sizes = {
        sp.name: len(vals) for sp, vals in zip(sweep_params, sweep_values)
    }
    assert xr_ds.sizes == expected_sizes

    # Check coords contents and order
    for sp, vals in zip(sweep_params, sweep_values):
        assert sp.name in xr_ds.coords
        np.testing.assert_allclose(xr_ds.coords[sp.name].values, vals)

    # Check measured data dims and values
    assert "signal" in xr_ds.data_vars
    expected_dims = tuple(sp.name for sp in sweep_params)
    assert xr_ds["signal"].dims == expected_dims

    # Build expected grid via meshgrid and compare
    grids = np.meshgrid(*sweep_values, indexing="ij")
    expected_signal = np.zeros(tuple(points_per_dim), dtype=float)
    for i, grid in enumerate(grids):
        expected_signal += weights[i] * grid.astype(float)

    np.testing.assert_allclose(xr_ds["signal"].values, expected_signal)


def test_netcdf_export_with_none_timestamp_raw(
    tmp_path_factory: TempPathFactory, experiment: Experiment
) -> None:
    """
    Test that datasets with None timestamp_raw values export correctly to NetCDF
    using sentinel values and import back with correct None values.
    """
    tmp_path = tmp_path_factory.mktemp("netcdf_none_timestamp")

    # Create a dataset that will have None timestamp_raw values
    # Don't prepare it or add data, just like test_write_metadata_to_explicit_db
    ds = DataSetInMem._create_new_run(name="test_none_timestamp")

    # Verify initial state - both timestamp_raw should be None since we didn't start or complete
    assert ds.run_timestamp_raw is None
    assert ds.completed_timestamp_raw is None

    # Export to NetCDF directly without preparing or adding data
    file_path = tmp_path / f"test_{ds.captured_run_id}_{ds.guid}.nc"
    ds.export(export_type="netcdf", path=str(tmp_path), prefix="test_")

    # Verify the file was created
    assert file_path.exists()

    # Load the raw NetCDF file to check sentinel values are used
    with xr.open_dataset(file_path, engine="h5netcdf") as loaded_xr:
        # Check that sentinel values (-1) are present in the NetCDF file
        assert loaded_xr.attrs["run_timestamp_raw"] == -1
        assert loaded_xr.attrs["completed_timestamp_raw"] == -1

    # Load back through QCoDeS to verify sentinel conversion
    loaded_ds = DataSetInMem._load_from_netcdf(file_path)

    # Verify that sentinel values were converted back to None
    assert loaded_ds.run_timestamp_raw is None
    assert loaded_ds.completed_timestamp_raw is None

    # Verify other metadata is preserved
    assert loaded_ds.captured_run_id == ds.captured_run_id
    assert loaded_ds.guid == ds.guid
    assert loaded_ds.name == ds.name


def test_netcdf_export_with_mixed_timestamp_raw(
    tmp_path_factory: TempPathFactory, experiment: Experiment
) -> None:
    """
    Test NetCDF export/import with one timestamp_raw being None and one being set.
    """
    tmp_path = tmp_path_factory.mktemp("netcdf_mixed_timestamp")

    # Create a dataset and prepare it (this sets run_timestamp_raw)
    ds = DataSetInMem._create_new_run(name="test_mixed_timestamp")

    # Add some minimal data
    x_param = ParamSpecBase("x", paramtype="numeric")
    y_param = ParamSpecBase("y", paramtype="numeric")

    interdeps = InterDependencies_(
        dependencies={y_param: (x_param,)}, inferences={}, standalones=()
    )
    ds.prepare(interdeps=interdeps, snapshot={})

    # Add a data point
    ds._enqueue_results({x_param: np.array([1.0]), y_param: np.array([2.0])})

    # Verify run_timestamp_raw is set but completed_timestamp_raw is None
    # (because we didn't call mark_completed())
    assert ds.run_timestamp_raw is not None
    assert ds.completed_timestamp_raw is None

    # Export without completing (so completed_timestamp_raw stays None)
    file_path = tmp_path / f"test_{ds.captured_run_id}_{ds.guid}.nc"
    ds.export(export_type="netcdf", path=str(tmp_path), prefix="test_")

    # Check raw NetCDF file
    with xr.open_dataset(file_path, engine="h5netcdf") as loaded_xr:
        # run_timestamp_raw should be the actual timestamp (not -1)
        assert loaded_xr.attrs["run_timestamp_raw"] != -1
        assert loaded_xr.attrs["run_timestamp_raw"] == ds.run_timestamp_raw
        # completed_timestamp_raw should be sentinel value (-1)
        assert loaded_xr.attrs["completed_timestamp_raw"] == -1

    # Load back and verify conversion
    loaded_ds = DataSetInMem._load_from_netcdf(file_path)

    # Verify timestamp_raw values are correct
    assert loaded_ds.run_timestamp_raw == ds.run_timestamp_raw
    assert loaded_ds.completed_timestamp_raw is None


@given(data=hst.data())
@settings(
    max_examples=10,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
    deadline=None,
)
def test_measurement_hypothesis_nd_grid_with_inferred_param(
    data: hst.DataObject, experiment: Experiment, caplog: LogCaptureFixture
) -> None:
    """
    Randomized ND sweep using Measurement context manager with an inferred parameter:
    - Draw N in [2, 4]
    - For each dimension i, draw number of points n_i in [1, 5]
    - Sweep each ManualParameter over a linspace of length n_i
    - Choose m in [1, N-1] and a subset of m swept parameters for an inferred coord
    - Register an inferred parameter depending on that subset and add its values
    - Measure a deterministic function of the setpoints
    - Assert xarray dims, coords (including inferred), and data match expectation
    """
    # number of dimensions and points per dimension
    n_dims = data.draw(hst.integers(min_value=2, max_value=4), label="n_dims")
    points_per_dim = [
        data.draw(hst.integers(min_value=1, max_value=5), label=f"n_points_dim_{i}")
        for i in range(n_dims)
    ]

    # build setpoint arrays and names
    sp_names = [f"x{i}" for i in range(n_dims)]
    sp_values: list[np.ndarray] = [
        np.linspace(0.0, float(npts - 1), npts) for npts in points_per_dim
    ]

    # choose subset for inferred parameter (strict subset)
    m = data.draw(hst.integers(min_value=1, max_value=n_dims - 1), label="m")
    inf_indices = sorted(
        data.draw(
            hst.lists(
                hst.integers(min_value=0, max_value=n_dims - 1),
                min_size=m,
                max_size=m,
                unique=True,
            ),
            label="inf_indices",
        )
    )
    inf_sp_names = [sp_names[i] for i in inf_indices]

    # weights for measured signal
    weights = [(i + 1) for i in range(n_dims)]

    # Setup measurement with shapes so xarray direct path is used
    meas = Measurement(exp=experiment, name="nd_grid_with_inferred")
    # register setpoints
    for name in sp_names:
        meas.register_custom_parameter(name, paramtype="numeric")
    # register inferred parameter (from subset of setpoints)
    meas.register_custom_parameter(
        "inf", basis=tuple(inf_sp_names), paramtype="numeric"
    )
    # register measured parameter depending on all setpoints
    meas.register_custom_parameter(
        "signal", setpoints=tuple(sp_names), paramtype="numeric"
    )
    meas.set_shapes({"signal": tuple(points_per_dim)})

    # run measurement over full grid
    with meas.run() as datasaver:
        # iterate over grid indices
        for idx in np.ndindex(*points_per_dim):
            # collect setpoint values for this point
            sp_items: list[tuple[str, float]] = [
                (sp_names[k], float(sp_values[k][idx[k]])) for k in range(n_dims)
            ]
            # measured signal: weighted sum of all setpoints
            signal_val = float(
                sum(weights[k] * float(sp_values[k][idx[k]]) for k in range(n_dims))
            )
            # inferred value: sum over selected subset of setpoints
            inf_val = float(sum(float(sp_values[k][idx[k]]) for k in inf_indices))
            results: list[tuple[str, float]] = [
                *sp_items,
                ("inf", inf_val),
                ("signal", signal_val),
            ]
            datasaver.add_result(*results)

    ds = datasaver.dataset

    # export to xarray and ensure direct path used
    caplog.clear()
    with caplog.at_level(logging.INFO):
        xr_ds = ds.to_xarray_dataset()

    assert any(
        "Exporting signal to xarray using direct method" in record.message
        for record in caplog.records
    )

    # Expected sizes per coordinate (all setpoints)
    expected_sizes = {name: len(vals) for name, vals in zip(sp_names, sp_values)}
    assert xr_ds.sizes == expected_sizes

    # Check setpoint coords contents and order
    for name, vals in zip(sp_names, sp_values):
        assert name in xr_ds.coords
        np.testing.assert_allclose(xr_ds.coords[name].values, vals)

    # Measured data dims and values
    assert "signal" in xr_ds.data_vars
    assert xr_ds["signal"].dims == tuple(sp_names)

    grids_all = np.meshgrid(*sp_values, indexing="ij")
    expected_signal = np.zeros(tuple(points_per_dim), dtype=float)
    for i, grid in enumerate(grids_all):
        expected_signal += weights[i] * grid.astype(float)
    np.testing.assert_allclose(xr_ds["signal"].values, expected_signal)

    # Inferred coord should be present with dims equal to the subset order
    assert "inf" in xr_ds.coords
    expected_inf_dims = tuple(inf_sp_names)
    assert xr_ds.coords["inf"].dims == expected_inf_dims

    # Build expected inferred grid based only on the subset dims
    subset_values = [sp_values[i] for i in inf_indices]
    grids_subset = np.meshgrid(*subset_values, indexing="ij") if subset_values else []
    expected_inf = np.zeros(tuple(points_per_dim[i] for i in inf_indices), dtype=float)
    for grid in grids_subset:
        expected_inf += grid.astype(float)
    np.testing.assert_allclose(xr_ds.coords["inf"].values, expected_inf)

    # The indexes of the inferred coord must correspond to the axes it depends on
    # i.e., keys should match the inferred-from setpoint names, and each index equal
    # to the dataset's index for that dimension
    inf_indexes = xr_ds.coords["inf"].indexes
    assert set(inf_indexes.keys()) == set(inf_sp_names)
    for dim in inf_sp_names:
        assert inf_indexes[dim].equals(xr_ds.indexes[dim])


def test_measurement_2d_with_inferred_setpoint(
    experiment: Experiment, caplog: LogCaptureFixture
) -> None:
    """
    Sweep two parameters (x, y) where y is inferred from one or more basis parameters.
    Verify that xarray export uses direct method, signal dims match, and basis
    parameters appear as inferred coordinates with indexes corresponding to y.
    """
    # Grid sizes
    nx, ny = 3, 4
    x_vals = np.linspace(0.0, 2.0, nx)
    # Define basis parameters for y and compute y from these
    y_b0_vals = np.linspace(10.0, 13.0, ny)
    y_b1_vals = np.linspace(-1.0, 2.0, ny)
    # y is inferred from (y_b0, y_b1)
    y_vals = y_b0_vals + 2.0 * y_b1_vals

    meas = Measurement(exp=experiment, name="2d_with_inferred_setpoint")
    # Register setpoint x
    meas.register_custom_parameter("x", paramtype="numeric")
    # Register basis params for y
    meas.register_custom_parameter("y_b0", paramtype="numeric")
    meas.register_custom_parameter("y_b1", paramtype="numeric")
    # Register y as setpoint inferred from basis
    meas.register_custom_parameter("y", basis=("y_b0", "y_b1"), paramtype="numeric")
    # Register measured parameter depending on (x, y)
    meas.register_custom_parameter("signal", setpoints=("x", "y"), paramtype="numeric")
    meas.set_shapes({"signal": (nx, ny)})

    with meas.run() as datasaver:
        for ix in range(nx):
            for iy in range(ny):
                x = float(x_vals[ix])
                y_b0 = float(y_b0_vals[iy])
                y_b1 = float(y_b1_vals[iy])
                y = float(y_vals[iy])
                signal = x + 3.0 * y  # deterministic function
                datasaver.add_result(
                    ("x", x),
                    ("y_b0", y_b0),
                    ("y_b1", y_b1),
                    ("y", y),
                    ("signal", signal),
                )

    ds = datasaver.dataset

    caplog.clear()
    with caplog.at_level(logging.INFO):
        xr_ds = ds.to_xarray_dataset()

    assert any(
        "Exporting signal to xarray using direct method" in record.message
        for record in caplog.records
    )

    # Sizes and coords
    assert xr_ds.sizes == {"x": nx, "y": ny}
    np.testing.assert_allclose(xr_ds.coords["x"].values, x_vals)
    np.testing.assert_allclose(xr_ds.coords["y"].values, y_vals)

    # Signal dims and values
    assert xr_ds["signal"].dims == ("x", "y")
    expected_signal = x_vals[:, None] + 3.0 * y_vals[None, :]
    np.testing.assert_allclose(xr_ds["signal"].values, expected_signal)

    # Inferred coords for y_b0 and y_b1 exist with dims only along y
    for name, vals in ("y_b0", y_b0_vals), ("y_b1", y_b1_vals):
        assert name in xr_ds.coords
        assert xr_ds.coords[name].dims == ("y",)
        np.testing.assert_allclose(xr_ds.coords[name].values, vals)
        # Indexes of inferred coords should correspond to the y axis index
        inf_idx = xr_ds.coords[name].indexes
        assert set(inf_idx.keys()) == {"y"}
        assert inf_idx["y"].equals(xr_ds.indexes["y"])


def test_measurement_2d_with_inferred_setpoint_from_setpoint(
    experiment: Experiment, caplog: LogCaptureFixture
) -> None:
    """
    This is not a good idea but a user can do this
    """
    # Grid sizes
    nx, ny = 3, 4
    x_vals = np.linspace(0.0, 2.0, nx)
    y_vals = np.linspace(10.0, 13.0, ny)

    meas = Measurement(exp=experiment, name="2d_with_inferred_setpoint")
    # Register setpoint x
    meas.register_custom_parameter("x", paramtype="numeric")

    # Register y as setpoint inferred from basis
    meas.register_custom_parameter("y", basis=("x"), paramtype="numeric")
    # Register measured parameter depending on (x, y)
    meas.register_custom_parameter("signal", setpoints=("x", "y"), paramtype="numeric")
    meas.set_shapes({"signal": (nx, ny)})

    with meas.run() as datasaver:
        for ix in range(nx):
            for iy in range(ny):
                x = float(x_vals[ix])
                y = float(y_vals[iy])
                signal = x + 3.0 * y  # deterministic function
                datasaver.add_result(
                    ("x", x),
                    ("y", y),
                    ("signal", signal),
                )

    ds = datasaver.dataset

    caplog.clear()
    with caplog.at_level(logging.INFO):
        xr_ds = ds.to_xarray_dataset()

    assert any(
        "Exporting signal to xarray using direct method" in record.message
        for record in caplog.records
    )

    # Sizes and coords
    assert xr_ds.sizes == {"x": nx, "y": ny}
    np.testing.assert_allclose(xr_ds.coords["x"].values, x_vals)
    np.testing.assert_allclose(xr_ds.coords["y"].values, y_vals)

    assert len(xr_ds.coords) == 2

    # Signal dims and values
    assert xr_ds["signal"].dims == ("x", "y")
    expected_signal = x_vals[:, None] + 3.0 * y_vals[None, :]
    np.testing.assert_allclose(xr_ds["signal"].values, expected_signal)


def test_measurement_2d_top_level_inferred_is_data_var(
    experiment: Experiment, caplog: LogCaptureFixture
) -> None:
    """
    If an inferred parameter is related to the top-level measured parameter,
    it must be exported as a data variable (not a coordinate) with the full
    dependency dimensions.
    """
    nx, ny = 2, 3
    x_vals = np.linspace(0.0, 1.0, nx)
    y_vals = np.linspace(10.0, 12.0, ny)

    # Define a measured signal and an inferred param both defined on (x, y)
    # The inferred param is related to the measured top-level param in the graph
    meas = Measurement(exp=experiment, name="2d_top_level_inferred")
    meas.register_custom_parameter("x", paramtype="numeric")
    meas.register_custom_parameter("y", paramtype="numeric")
    # Register measured top-level
    meas.register_custom_parameter("signal", setpoints=("x", "y"), paramtype="numeric")
    # Register inferred related to top-level (basis includes the measured top-level)
    meas.register_custom_parameter("derived", basis=("signal",), paramtype="numeric")
    meas.set_shapes({"signal": (nx, ny)})

    with meas.run() as datasaver:
        for ix in range(nx):
            for iy in range(ny):
                x = float(x_vals[ix])
                y = float(y_vals[iy])
                signal = x + y
                derived = 2.0 * signal  # inferred from top-level
                datasaver.add_result(
                    ("x", x), ("y", y), ("signal", signal), ("derived", derived)
                )

    ds = datasaver.dataset
    caplog.clear()
    with caplog.at_level(logging.INFO):
        xr_ds = ds.to_xarray_dataset()

    # Direct path log should be present
    assert any(
        "Exporting signal to xarray using direct method" in record.message
        for record in caplog.records
    )

    # The derived param should be a data variable with dims (x, y), not a coord
    assert "derived" in xr_ds.data_vars
    assert "derived" not in xr_ds.coords
    assert xr_ds["derived"].dims == ("x", "y")

    expected_signal = x_vals[:, None] + y_vals[None, :]
    expected_derived = 2.0 * expected_signal
    np.testing.assert_allclose(xr_ds["signal"].values, expected_signal)
    np.testing.assert_allclose(xr_ds["derived"].values, expected_derived)


def test_with_without_shape_is_the_same(experiment: Experiment) -> None:
    nx, ny = 2, 3
    x_vals = np.linspace(0.0, -1.0, nx)
    y_vals = np.linspace(10.0, 12.0, ny)

    # simple 2d grid with no shapes
    meas1 = Measurement(exp=experiment, name="2d_no_shape")
    meas1.register_custom_parameter("x", paramtype="numeric")
    meas1.register_custom_parameter("y", paramtype="numeric")
    meas1.register_custom_parameter("z", paramtype="numeric", setpoints=("x", "y"))
    with meas1.run() as datasaver:
        for ix in range(nx):
            for iy in range(ny):
                x = float(x_vals[ix])
                y = float(y_vals[iy])
                z = x + y
                datasaver.add_result(
                    ("x", x),
                    ("y", y),
                    ("z", z),
                )
        ds1 = datasaver.dataset

    dsx1 = ds1.to_xarray_dataset()

    # simple 2d grid with knwon shapes
    meas2 = Measurement(exp=experiment, name="2d_shape")
    meas2.register_custom_parameter("x", paramtype="numeric")
    meas2.register_custom_parameter("y", paramtype="numeric")
    meas2.register_custom_parameter("z", paramtype="numeric", setpoints=("x", "y"))
    meas2.set_shapes({"z": (nx, ny)})
    with meas2.run() as datasaver:
        for ix in range(nx):
            for iy in range(ny):
                x = float(x_vals[ix])
                y = float(y_vals[iy])
                z = x + y
                datasaver.add_result(
                    ("x", x),
                    ("y", y),
                    ("z", z),
                )
        ds2 = datasaver.dataset
    dsx2 = ds2.to_xarray_dataset()

    # the two export methods inverts the x axis such that the new export method
    # matches the order the data is written in which is arguably more correct
    assert_array_equal(dsx2["x"].values, x_vals)

    assert_array_equal(np.flip(dsx1["x"].values), dsx2["x"].values)

    dsx2_sorted = dsx2.sortby(["x", "y"])

    assert_array_equal(dsx1["x"].values, dsx2_sorted["x"].values)

    # however data for a given coordinate is the same
    assert bool((dsx2 - dsx1)["z"].max() == 0)
    assert bool((dsx2 - dsx1)["x"].max() == 0)
