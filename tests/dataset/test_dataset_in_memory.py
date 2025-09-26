import contextlib
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import hypothesis.strategies as hst
import numpy as np
import pytest
import xarray as xr
from deepdiff import DeepDiff  # type: ignore[import-untyped]
from hypothesis import HealthCheck, given, settings
from numpy.testing import assert_almost_equal

import qcodes
from qcodes.dataset import Measurement, load_by_id, load_by_run_spec
from qcodes.dataset.data_set_in_memory import DataSetInMem, load_from_file
from qcodes.dataset.data_set_protocol import DataSetType
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.sqlite.connection import AtomicConnection, atomic_transaction
from qcodes.parameters import ManualParameter, Parameter, ParamSpecBase
from qcodes.station import Station

if TYPE_CHECKING:
    from qcodes.dataset.experiment_container import Experiment


def test_dataset_in_memory_reload_from_db(
    meas_with_registered_param, DMM, DAC, tmp_path
) -> None:
    with meas_with_registered_param.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    ds = datasaver.dataset
    ds.add_metadata("mymetadatatag", 42)

    paramspecs = ds.get_parameters()
    assert len(paramspecs) == 2
    assert paramspecs[0].name == "dummy_dac_ch1"
    assert paramspecs[1].name == "dummy_dmm_v1"
    ds.export(export_type="netcdf", path=str(tmp_path))

    assert isinstance(ds, DataSetInMem)

    loaded_ds = load_by_id(ds.run_id)
    assert isinstance(loaded_ds, DataSetInMem)
    compare_datasets(ds, loaded_ds)


@settings(
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
    max_examples=10,
)
@given(
    shape1=hst.integers(min_value=1, max_value=100),
    shape2=hst.integers(min_value=1, max_value=100),
)
def test_dataset_in_memory_reload_from_db_2d(
    meas_with_registered_param_2d, DMM, DAC, tmp_path, shape1, shape2
) -> None:
    meas_with_registered_param_2d.set_shapes(
        {
            DMM.v1.full_name: (shape1, shape2),
        }
    )
    i = 0
    with meas_with_registered_param_2d.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, shape1):
            for set_v2 in np.linspace(0, 100, shape2):
                DAC.ch1.set(set_v)
                DAC.ch2.set(set_v2)
                datasaver.add_result(
                    (DAC.ch1, set_v), (DAC.ch2, set_v2), (DMM.v1, float(i))
                )
                i = i + 1
    ds = datasaver.dataset
    ds.add_metadata("mymetadatatag", 42)

    paramspecs = ds.get_parameters()
    assert len(paramspecs) == 3
    assert paramspecs[0].name == "dummy_dac_ch1"
    assert paramspecs[1].name == "dummy_dac_ch2"
    assert paramspecs[2].name == "dummy_dmm_v1"

    # if the indexes (their order) are not correct here, the exported xarray, and thus
    # the exported netcdf will have a wrong order of axes in the data, so that
    # the loaded data will have the coordinates inverted. Hence we assert that
    # the order is exactly the same as declared via Measurement.register_parameter
    # calls above
    assert tuple(ds.cache.to_pandas_dataframe().index.names) == (
        "dummy_dac_ch1",
        "dummy_dac_ch2",
    )

    ds.export(export_type="netcdf", path=str(tmp_path))

    assert isinstance(ds, DataSetInMem)

    loaded_ds = load_by_id(ds.run_id)
    assert isinstance(loaded_ds, DataSetInMem)
    compare_datasets(ds, loaded_ds)


@settings(
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
    max_examples=10,
)
@given(
    shape1=hst.integers(min_value=1, max_value=10),
    shape2=hst.integers(min_value=1, max_value=10),
    shape3=hst.integers(min_value=1, max_value=10),
)
def test_dataset_in_memory_reload_from_db_3d(
    meas_with_registered_param_3d, DMM, DAC3D, tmp_path, shape1, shape2, shape3
) -> None:
    meas_with_registered_param_3d.set_shapes(
        {
            DMM.v1.full_name: (shape1, shape2, shape3),
        }
    )
    i = 0
    with meas_with_registered_param_3d.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, shape1):
            for set_v2 in np.linspace(0, 100, shape2):
                for set_v3 in np.linspace(0, 400, shape3):
                    DAC3D.ch1.set(set_v)
                    DAC3D.ch2.set(set_v2)
                    DAC3D.ch3.set(set_v3)
                    datasaver.add_result(
                        (DAC3D.ch1, set_v),
                        (DAC3D.ch2, set_v2),
                        (DAC3D.ch3, set_v3),
                        (DMM.v1, float(i)),
                    )
                    i = i + 1
    ds = datasaver.dataset
    ds.add_metadata("mymetadatatag", 42)

    paramspecs = ds.get_parameters()
    assert len(paramspecs) == 4
    assert paramspecs[0].name == "dummy_dac_ch1"
    assert paramspecs[1].name == "dummy_dac_ch2"
    assert paramspecs[2].name == "dummy_dac_ch3"
    assert paramspecs[3].name == "dummy_dmm_v1"

    # if the indexes (their order) are not correct here, the exported xarray, and thus
    # the exported netcdf will have a wrong order of axes in the data, so that
    # the loaded data will have the coordinates inverted. Hence we assert that
    # the order is exactly the same as declared via Measurement.register_parameter
    # calls above
    assert tuple(ds.cache.to_pandas_dataframe().index.names) == (
        "dummy_dac_ch1",
        "dummy_dac_ch2",
        "dummy_dac_ch3",
    )

    ds.export(export_type="netcdf", path=str(tmp_path))

    assert isinstance(ds, DataSetInMem)

    loaded_ds = load_by_id(ds.run_id)
    assert isinstance(loaded_ds, DataSetInMem)
    compare_datasets(ds, loaded_ds)


def test_dataset_in_memory_without_cache_raises(
    meas_with_registered_param, DMM, DAC, tmp_path
) -> None:
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Cannot disable the in memory cache for a dataset that is only in memory."
        ),
    ):
        with meas_with_registered_param.run(
            dataset_class=DataSetType.DataSetInMem, in_memory_cache=False
        ) as datasaver:
            for set_v in np.linspace(0, 25, 10):
                DAC.ch1.set(set_v)
                get_v = DMM.v1()
                datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))


def test_dataset_in_memory_reload_from_db_complex(
    meas_with_registered_param_complex, DAC, complex_num_instrument, tmp_path
) -> None:
    with meas_with_registered_param_complex.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = complex_num_instrument.complex_num()
            datasaver.add_result(
                (DAC.ch1, set_v), (complex_num_instrument.complex_num, get_v)
            )

    ds = datasaver.dataset
    ds.add_metadata("mymetadatatag", 42)
    ds.export(export_type="netcdf", path=str(tmp_path))

    assert isinstance(ds, DataSetInMem)

    loaded_ds = load_by_id(ds.run_id)
    assert isinstance(loaded_ds, DataSetInMem)
    compare_datasets(ds, loaded_ds)


def test_dataset_in_memory_reload_from_netcdf_complex(
    meas_with_registered_param_complex, DAC, complex_num_instrument, tmp_path
) -> None:
    with meas_with_registered_param_complex.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = complex_num_instrument.complex_num()
            datasaver.add_result(
                (DAC.ch1, set_v), (complex_num_instrument.complex_num, get_v)
            )

    ds = datasaver.dataset
    ds.add_metadata("mymetadatatag", 42)
    ds.add_metadata("someothermetadatatag", 42)
    ds.export(export_type="netcdf", path=str(tmp_path))

    assert isinstance(ds, DataSetInMem)
    loaded_ds = DataSetInMem._load_from_netcdf(
        tmp_path / f"qcodes_{ds.captured_run_id}_{ds.guid}.nc"
    )
    assert isinstance(loaded_ds, DataSetInMem)
    compare_datasets(ds, loaded_ds)


def test_dataset_in_memory_no_export_warns(
    meas_with_registered_param, DMM, DAC, tmp_path
) -> None:
    with meas_with_registered_param.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    ds = datasaver.dataset
    ds.add_metadata("mymetadatatag", 42)
    assert isinstance(ds, DataSetInMem)
    ds.export(export_type="netcdf", path=str(tmp_path))
    os.remove(ds.export_info.export_paths["nc"])

    with pytest.warns(
        UserWarning, match="Could not load raw data for dataset with guid"
    ):
        loaded_ds = load_by_id(ds.run_id)

    assert isinstance(loaded_ds, DataSetInMem)

    assert loaded_ds.cache.data() == {}


def test_dataset_in_memory_missing_file_warns(
    meas_with_registered_param, DMM, DAC, tmp_path
) -> None:
    with meas_with_registered_param.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    ds = datasaver.dataset
    ds.add_metadata("mymetadatatag", 42)

    assert isinstance(ds, DataSetInMem)

    with pytest.warns(UserWarning, match="No raw data stored for dataset with guid"):
        loaded_ds = load_by_id(ds.run_id)
    assert isinstance(loaded_ds, DataSetInMem)

    assert loaded_ds.cache.data() == {}


def test_dataset_in_reload_from_netcdf(
    meas_with_registered_param, DMM, DAC, tmp_path
) -> None:
    with meas_with_registered_param.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    ds = datasaver.dataset
    ds.add_metadata("mymetadatatag", 42)
    assert isinstance(ds, DataSetInMem)

    ds.export(export_type="netcdf", path=str(tmp_path))

    ds.add_metadata("metadata_added_after_export", 69)

    loaded_ds = DataSetInMem._load_from_netcdf(
        tmp_path / f"qcodes_{ds.captured_run_id}_{ds.guid}.nc"
    )
    assert isinstance(loaded_ds, DataSetInMem)
    compare_datasets(ds, loaded_ds)


def test_dataset_load_from_netcdf_and_db(
    meas_with_registered_param, DMM, DAC, tmp_path
) -> None:
    with meas_with_registered_param.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    with meas_with_registered_param.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    path_to_db = datasaver.dataset._path_to_db
    ds = datasaver.dataset
    ds.add_metadata("mymetadatatag", 42)

    assert ds.run_id == 2
    assert isinstance(ds, DataSetInMem)

    ds.export(export_type="netcdf", path=str(tmp_path))

    ds.add_metadata("metadata_added_after_export", 69)

    loaded_ds = DataSetInMem._load_from_netcdf(
        tmp_path / f"qcodes_{ds.captured_run_id}_{ds.guid}.nc", path_to_db=path_to_db
    )
    assert isinstance(loaded_ds, DataSetInMem)
    assert loaded_ds.run_id == ds.run_id
    compare_datasets(ds, loaded_ds)


def test_dataset_in_memory_does_not_create_runs_table(
    meas_with_registered_param, DMM, DAC, tmp_path
) -> None:
    with meas_with_registered_param.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    ds = datasaver.dataset
    dbfile = datasaver.dataset._path_to_db

    conn = AtomicConnection(dbfile)

    tables_query = 'SELECT * FROM sqlite_master WHERE TYPE = "table"'
    tables = list(atomic_transaction(conn, tables_query).fetchall())
    assert len(tables) == 4
    tablenames = tuple(table[1] for table in tables)
    assert all(ds.name not in table_name for table_name in tablenames)


def test_load_from_netcdf_and_write_metadata_to_db(empty_temp_db) -> None:
    netcdf_file_path = (
        Path(__file__).parent / "fixtures" / "db_files" / "netcdf" / "qcodes_2.nc"
    )

    if not os.path.exists(str(netcdf_file_path)):
        pytest.skip("No netcdf fixtures found.")

    ds = DataSetInMem._load_from_netcdf(netcdf_file_path)
    ds.write_metadata_to_db()

    loaded_ds = load_by_run_spec(captured_run_id=ds.captured_run_id)
    assert isinstance(loaded_ds, DataSetInMem)
    assert loaded_ds.captured_run_id == ds.captured_run_id
    assert loaded_ds.captured_counter == ds.captured_counter
    assert loaded_ds.run_timestamp_raw == ds.run_timestamp_raw
    assert loaded_ds.completed_timestamp_raw == ds.completed_timestamp_raw

    compare_datasets(ds, loaded_ds)

    # now we attempt to write again. This should be a noop so everything should
    # stay the same
    ds.write_metadata_to_db()
    loaded_ds = load_by_run_spec(captured_run_id=ds.captured_run_id)
    assert isinstance(loaded_ds, DataSetInMem)
    assert loaded_ds.captured_run_id == ds.captured_run_id
    assert loaded_ds.captured_counter == ds.captured_counter
    assert loaded_ds.run_timestamp_raw == ds.run_timestamp_raw
    assert loaded_ds.completed_timestamp_raw == ds.completed_timestamp_raw

    compare_datasets(ds, loaded_ds)


def test_load_from_netcdf_no_db_file(non_created_db) -> None:
    netcdf_file_path = (
        Path(__file__).parent / "fixtures" / "db_files" / "netcdf" / "qcodes_2.nc"
    )

    if not os.path.exists(str(netcdf_file_path)):
        pytest.skip("No netcdf fixtures found.")

    ds = DataSetInMem._load_from_netcdf(netcdf_file_path)
    ds.write_metadata_to_db()
    loaded_ds = load_by_run_spec(captured_run_id=ds.captured_run_id)
    assert isinstance(loaded_ds, DataSetInMem)
    compare_datasets(ds, loaded_ds)


def test_load_from_db(meas_with_registered_param, DMM, DAC, tmp_path) -> None:
    Station(DAC, DMM)
    with meas_with_registered_param.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    ds = datasaver.dataset
    ds.add_metadata("foo", "bar")
    ds.export(export_type="netcdf", path=tmp_path)

    ds.add_metadata("metadata_added_after_export", 69)

    loaded_ds = load_by_id(ds.run_id)
    assert isinstance(loaded_ds, DataSetInMem)
    assert loaded_ds.snapshot == ds.snapshot
    assert loaded_ds.export_info == ds.export_info
    assert loaded_ds.metadata == ds.metadata

    assert "foo" in loaded_ds.metadata.keys()
    assert "export_info" in loaded_ds.metadata.keys()
    assert "metadata_added_after_export" in loaded_ds.metadata.keys()
    assert loaded_ds.metadata["metadata_added_after_export"] == 69

    compare_datasets(ds, loaded_ds)


def test_load_from_file(meas_with_registered_param, DMM, DAC, tmp_path) -> None:
    qcodes.config["dataset"]["export_prefix"] = "my_export_prefix"
    qcodes.config["dataset"]["export_type"] = "netcdf"

    Station(DAC, DMM)
    with meas_with_registered_param.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    ds: DataSetInMem = datasaver.dataset
    ds.add_metadata("foo", "bar")
    ds.export(path=tmp_path)
    ds.add_metadata("metadata_added_after_export", "42")

    export_file_path = ds.export_info.export_paths.get("nc")
    assert export_file_path is not None
    loaded_ds: DataSetInMem = load_from_file(export_file_path)

    assert isinstance(loaded_ds, DataSetInMem)
    assert loaded_ds.snapshot == ds.snapshot
    assert loaded_ds.export_info == ds.export_info
    assert loaded_ds.metadata == ds.metadata

    assert "export_info" in loaded_ds.metadata.keys()
    assert "metadata_added_after_export" in loaded_ds.metadata.keys()
    assert loaded_ds.metadata["foo"] == "bar"
    assert loaded_ds.metadata["metadata_added_after_export"] == "42"

    compare_datasets(ds, loaded_ds)


def test_load_from_file_by_id(meas_with_registered_param, DMM, DAC, tmp_path) -> None:
    qcodes.config["dataset"]["export_prefix"] = "my_export_prefix"
    qcodes.config["dataset"]["export_type"] = "netcdf"
    qcodes.config["dataset"]["load_from_exported_file"] = True
    assert qcodes.config.dataset.load_from_exported_file is True

    Station(DAC, DMM)
    with meas_with_registered_param.run() as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    ds: DataSetInMem = datasaver.dataset
    assert not isinstance(ds, DataSetInMem)
    ds.export(path=tmp_path)

    # Load from file
    loaded_ds_from_file = load_by_id(ds.run_id)
    assert isinstance(loaded_ds_from_file, DataSetInMem)

    # Load from db
    qcodes.config["dataset"]["load_from_exported_file"] = False
    assert qcodes.config.dataset.load_from_exported_file is False
    loaded_ds_from_db = load_by_id(ds.run_id)
    assert not isinstance(loaded_ds_from_db, DataSetInMem)


def test_load_from_netcdf_non_completed_dataset(experiment, tmp_path) -> None:
    """Test that non-completed datasets can be loaded from netcdf files."""
    # Create a non-completed dataset by NOT using the measurement context manager
    # which automatically completes the dataset on exit
    ds = DataSetInMem._create_new_run(name="test-dataset")

    # Set up interdependencies with simple parameters following the established pattern
    x_param = ParamSpecBase("x", paramtype="numeric")
    y_param = ParamSpecBase("y", paramtype="numeric")
    idps = InterDependencies_(dependencies={y_param: (x_param,)})
    ds.prepare(interdeps=idps, snapshot={})

    # Add some data points
    for x_val in np.linspace(0, 25, 5):
        y_val = x_val**2  # simple function
        ds._enqueue_results({x_param: np.array([x_val]), y_param: np.array([y_val])})

    # Note: do NOT call ds.mark_completed() to keep it non-completed

    # Verify that the dataset is not completed
    assert ds.completed_timestamp_raw is None
    assert not ds.completed

    # Export the non-completed dataset to NetCDF
    ds.export(export_type="netcdf", path=str(tmp_path))

    # Load the dataset from NetCDF
    loaded_ds = DataSetInMem._load_from_netcdf(
        tmp_path / f"qcodes_{ds.captured_run_id}_{ds.guid}.nc"
    )

    # Verify that the loaded dataset is still non-completed
    assert isinstance(loaded_ds, DataSetInMem)
    assert loaded_ds.completed_timestamp_raw is None
    assert not loaded_ds.completed

    # Compare other properties
    assert loaded_ds.captured_run_id == ds.captured_run_id
    assert loaded_ds.guid == ds.guid
    assert loaded_ds.name == ds.name
    assert loaded_ds.run_timestamp_raw == ds.run_timestamp_raw


def test_load_from_netcdf_legacy_version(non_created_db) -> None:
    # Qcodes 0.26 exported netcdf files did not contain
    # the parent dataset links and used a different engine to write data
    # check that it still loads correctly

    netcdf_file_path = (
        Path(__file__).parent / "fixtures" / "db_files" / "netcdf" / "qcodes_v26.nc"
    )

    if not os.path.exists(str(netcdf_file_path)):
        pytest.skip("No netcdf fixtures found.")

    ds = DataSetInMem._load_from_netcdf(netcdf_file_path)
    ds.write_metadata_to_db()
    loaded_ds = load_by_run_spec(captured_run_id=ds.captured_run_id)
    assert isinstance(loaded_ds, DataSetInMem)
    compare_datasets(ds, loaded_ds)


def compare_datasets(ds, loaded_ds):
    assert ds.the_same_dataset_as(loaded_ds)
    assert len(ds) == len(loaded_ds)
    assert len(ds) != 0
    for outer_var, inner_dict in ds.cache.data().items():
        for inner_var, expected_data in inner_dict.items():
            assert (
                expected_data.shape
                == loaded_ds.cache.data()[outer_var][inner_var].shape
            )
            assert_almost_equal(
                expected_data,
                loaded_ds.cache.data()[outer_var][inner_var],
            )

    xds = ds.cache.to_xarray_dataset()
    loaded_xds = loaded_ds.cache.to_xarray_dataset()
    assert xds.sizes == loaded_xds.sizes
    assert all(xds == loaded_xds)


def test_load_from_db_dataset_moved(
    meas_with_registered_param, DMM, DAC, tmp_path
) -> None:
    Station(DAC, DMM)
    with meas_with_registered_param.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    ds = datasaver.dataset
    ds.add_metadata("foo", "bar")
    ds.export(export_type="netcdf", path=tmp_path)

    ds.add_metadata("metadata_added_after_export", 69)

    export_path = ds.export_info.export_paths["nc"]

    with contextlib.closing(xr.open_dataset(export_path)) as xr_ds:
        assert xr_ds.attrs["metadata_added_after_export"] == 69

    new_path = str(Path(export_path).parent / "someotherfilename.nc")
    shutil.move(export_path, new_path)

    with pytest.warns(
        UserWarning, match="Could not load raw data for dataset with guid"
    ):
        loaded_ds = load_by_id(ds.run_id)

    assert isinstance(loaded_ds, DataSetInMem)
    assert loaded_ds.snapshot == ds.snapshot
    assert loaded_ds.export_info == ds.export_info
    assert loaded_ds.metadata == ds.metadata

    assert "foo" in loaded_ds.metadata.keys()
    assert "export_info" in loaded_ds.metadata.keys()
    assert "metadata_added_after_export" in loaded_ds.metadata.keys()

    assert loaded_ds.cache.data() == {}

    with pytest.warns(
        UserWarning, match="Could not add metadata to the exported NetCDF file"
    ):
        ds.add_metadata("metadata_added_after_move", 696)

    with contextlib.closing(xr.open_dataset(new_path)) as new_xr_ds:
        assert new_xr_ds.attrs["metadata_added_after_export"] == 69
        assert "metadata_added_after_move" not in new_xr_ds.attrs

    loaded_ds.set_netcdf_location(new_path)

    assert loaded_ds.cache.data().keys() == ds.cache.data().keys()

    with contextlib.closing(xr.open_dataset(new_path)) as new_xr_ds:
        assert new_xr_ds.attrs["metadata_added_after_export"] == 69
        assert "metadata_added_after_move" not in new_xr_ds.attrs

    # This should have effect neither on the loaded_ds nor on the netcdf file
    with pytest.warns(
        UserWarning, match="Could not add metadata to the exported NetCDF file"
    ):
        ds.add_metadata(
            "metadata_added_to_old_dataset_after_set_new_netcdf_location", 696977
        )

    loaded_ds.add_metadata("metadata_added_after_set_new_netcdf_location", 6969)

    with contextlib.closing(xr.open_dataset(new_path)) as new_xr_ds:
        assert new_xr_ds.attrs["metadata_added_after_export"] == 69
        assert "metadata_added_after_move" not in new_xr_ds.attrs
        assert (
            "metadata_added_to_old_dataset_after_set_new_netcdf_location"
            not in new_xr_ds.attrs
        )
        assert new_xr_ds.attrs["metadata_added_after_set_new_netcdf_location"] == 6969


@pytest.mark.parametrize("include_inferred_data", [True, False])
def test_dataset_in_mem_with_inferred_parameters(
    experiment: "Experiment", include_inferred_data: bool
) -> None:
    inferred1 = ManualParameter("inferred1", initial_value=0.0)
    inferred2 = ManualParameter("inferred2", initial_value=0.0)
    control1 = ManualParameter("control1", initial_value=0.0)
    control2 = ManualParameter("control2", initial_value=0.0)
    dependent = Parameter("dependent", get_cmd=lambda: control1(), set_cmd=False)
    meas = Measurement(exp=experiment, name="via Measurement")

    meas.register_parameter(control1)
    meas.register_parameter(control2)
    meas.register_parameter(inferred1, basis=(control1, control2))
    meas.register_parameter(inferred2, basis=(control1, control2))
    meas.register_parameter(dependent, setpoints=(control1, control2))
    meas.set_shapes({dependent.register_name: (11, 11)})
    with meas.run() as datasaver:
        for i in range(11):
            for j in range(11):
                control1(float(i))
                control2(float(j))
                if include_inferred_data:
                    datasaver.add_result(
                        (inferred1, inferred1()),
                        (inferred2, inferred2()),
                        (control1, control1()),
                        (control2, control2()),
                        (dependent, dependent()),
                    )
                else:
                    datasaver.add_result(
                        (control1, control1()),
                        (control2, control2()),
                        (dependent, dependent()),
                    )
        ds = datasaver.dataset

    param_data = ds.get_parameter_data()
    cache_data = ds.cache.data()

    assert set(param_data.keys()) == set(cache_data.keys())
    assert set(param_data["dependent"].keys()) == set(cache_data["dependent"].keys())

    assert DeepDiff(param_data, cache_data, ignore_nan_inequality=True) == {}
