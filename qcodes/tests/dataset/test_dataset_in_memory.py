import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from qcodes import load_by_id
from qcodes.dataset import load_by_run_spec
from qcodes.dataset.data_set_in_memory import DataSetInMem
from qcodes.dataset.sqlite.connection import ConnectionPlus, atomic_transaction
from qcodes.station import Station


def test_dataset_in_memory_smoke_test(meas_with_registered_param, DMM, DAC, tmp_path):
    with meas_with_registered_param.run(dataset_class=DataSetInMem) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    ds = datasaver.dataset

    assert isinstance(ds, DataSetInMem)

    ds.export(export_type="netcdf", path=str(tmp_path))
    loaded_ds = DataSetInMem.load_from_netcdf(tmp_path / "qcodes_1.nc")
    assert isinstance(loaded_ds, DataSetInMem)
    compare_datasets(ds, loaded_ds)

    loaded_ds_2 = load_by_id(ds.run_id)
    assert isinstance(loaded_ds_2, DataSetInMem)
    compare_datasets(ds, loaded_ds_2)


def test_dataset_in_memory_does_not_create_runs_table(
    meas_with_registered_param, DMM, DAC, tmp_path
):
    with meas_with_registered_param.run(dataset_class=DataSetInMem) as datasaver:
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
        Path(__file__).parent / "fixtures" / "db_files" / "version8" / "qcodes_2.nc"
    )

    if not os.path.exists(str(netcdf_file_path)):
        pytest.skip("No netcdf fixtures found.")

    ds = DataSetInMem.load_from_netcdf(netcdf_file_path)
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
        Path(__file__).parent / "fixtures" / "db_files" / "version8" / "qcodes_2.nc"
    )

    if not os.path.exists(str(netcdf_file_path)):
        pytest.skip("No netcdf fixtures found.")

    ds = DataSetInMem.load_from_netcdf(netcdf_file_path)
    ds.write_metadata_to_db()
    loaded_ds = load_by_run_spec(captured_run_id=ds.captured_run_id)
    assert isinstance(loaded_ds, DataSetInMem)
    compare_datasets(ds, loaded_ds)


def test_load_from_db(meas_with_registered_param, DMM, DAC, tmp_path):
    Station(DAC, DMM)
    with meas_with_registered_param.run(dataset_class=DataSetInMem) as datasaver:
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
    # todo do we want this. e.g. should metadata contain
    # snapshot and export info even if this is accessible in other way
    assert "snapshot" in loaded_ds.metadata.keys()
    assert "export_info" in loaded_ds.metadata.keys()

    compare_datasets(ds, loaded_ds)


# todo missing from runs table
# snapshot, completed timestamp, parameters (do we care), verify other metadata
# When should metadata be added. In the old dataset it used to be added as
# soon as you call add_metadata


# add a test to import from 0.26 data (missing parent dataset links)


def compare_datasets(ds, loaded_ds):
    assert ds.the_same_dataset_as(loaded_ds)
    assert len(ds) == len(loaded_ds)
    assert len(ds) != 0
    xds = ds.cache.to_xarray_dataset()
    loaded_xds = loaded_ds.cache.to_xarray_dataset()
    assert xds.sizes == loaded_xds.sizes
    assert all(xds == loaded_xds)
