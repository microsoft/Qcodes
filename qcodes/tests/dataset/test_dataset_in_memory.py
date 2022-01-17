import os
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from qcodes import load_by_id
from qcodes.dataset import load_by_run_spec
from qcodes.dataset.data_set_in_memory import DataSetInMem
from qcodes.dataset.data_set_protocol import DataSetType
from qcodes.dataset.sqlite.connection import ConnectionPlus, atomic_transaction
from qcodes.station import Station


def test_dataset_in_memory_reload_from_db(
    meas_with_registered_param, DMM, DAC, tmp_path
):
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


def test_dataset_in_memory_without_cache_raises(
    meas_with_registered_param, DMM, DAC, tmp_path
):

    with pytest.raises(
        RuntimeError,
        match="Cannot disable the in memory cache for a dataset that is only in memory.",
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
):
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
):
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
):
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
):
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


def test_dataset_in_reload_from_netcdf(meas_with_registered_param, DMM, DAC, tmp_path):
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
    loaded_ds = DataSetInMem._load_from_netcdf(
        tmp_path / f"qcodes_{ds.captured_run_id}_{ds.guid}.nc"
    )
    assert isinstance(loaded_ds, DataSetInMem)
    compare_datasets(ds, loaded_ds)


def test_dataset_load_from_netcdf_and_db(
    meas_with_registered_param, DMM, DAC, tmp_path
):
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
    loaded_ds = DataSetInMem._load_from_netcdf(
        tmp_path / f"qcodes_{ds.captured_run_id}_{ds.guid}.nc", path_to_db=path_to_db
    )
    assert isinstance(loaded_ds, DataSetInMem)
    assert loaded_ds.run_id == ds.run_id
    compare_datasets(ds, loaded_ds)


def test_dataset_in_memory_does_not_create_runs_table(
    meas_with_registered_param, DMM, DAC, tmp_path
):
    with meas_with_registered_param.run(
        dataset_class=DataSetType.DataSetInMem
    ) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    ds = datasaver.dataset
    dbfile = datasaver.dataset._path_to_db

    conn = ConnectionPlus(sqlite3.connect(dbfile))

    tables_query = 'SELECT * FROM sqlite_master WHERE TYPE = "table"'
    tables = list(atomic_transaction(conn, tables_query).fetchall())
    assert len(tables) == 4
    tablenames = tuple(table[1] for table in tables)
    assert all(ds.name not in table_name for table_name in tablenames)


def test_load_from_netcdf_and_write_metadata_to_db(empty_temp_db):
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


def test_load_from_netcdf_no_db_file(non_created_db):
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


def test_load_from_db(meas_with_registered_param, DMM, DAC, tmp_path):
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
    loaded_ds = load_by_id(ds.run_id)
    assert isinstance(loaded_ds, DataSetInMem)
    assert loaded_ds.snapshot == ds.snapshot
    assert loaded_ds.export_info == ds.export_info
    assert loaded_ds.metadata == ds.metadata

    assert "foo" in loaded_ds.metadata.keys()
    assert "export_info" in loaded_ds.metadata.keys()

    compare_datasets(ds, loaded_ds)


def test_load_from_netcdf_legacy_version(non_created_db):
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
    xds = ds.cache.to_xarray_dataset()
    loaded_xds = loaded_ds.cache.to_xarray_dataset()
    assert xds.sizes == loaded_xds.sizes
    assert all(xds == loaded_xds)


def test_load_from_db_dataset_moved(meas_with_registered_param, DMM, DAC, tmp_path):
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

    export_path = ds.export_info.export_paths["nc"]
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
    assert loaded_ds.cache.data() == {}

    loaded_ds.set_netcdf_location(new_path)

    assert loaded_ds.cache.data().keys() == ds.cache.data().keys()
