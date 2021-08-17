import sqlite3

import numpy as np

from qcodes.dataset.data_set_in_memory import DataSetInMem
from qcodes.dataset.sqlite.connection import ConnectionPlus, atomic_transaction


def test_dataset_in_memory_smoke_test(meas_with_registered_param, DMM, DAC, tmp_path):
    with meas_with_registered_param.run(dataset_class=DataSetInMem) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    dataset = datasaver.dataset
    dataset.export(export_type="netcdf", path=tmp_path)
    loaded_ds = DataSetInMem.load_from_netcdf(tmp_path / "qcodes_1.nc")
    assert dataset.the_same_dataset_as(loaded_ds)


def test_dataset_in_memory_does_not_create_runs_table(
    meas_with_registered_param, DMM, DAC, tmp_path
):
    with meas_with_registered_param.run(dataset_class=DataSetInMem) as datasaver:
        for set_v in np.linspace(0, 25, 10):
            DAC.ch1.set(set_v)
            get_v = DMM.v1()
            datasaver.add_result((DAC.ch1, set_v), (DMM.v1, get_v))

    dataset = datasaver.dataset
    dbfile = datasaver.dataset._path_to_db

    conn = ConnectionPlus(sqlite3.connect(dbfile))

    tables_query = 'SELECT * FROM sqlite_master WHERE TYPE = "table"'
    tables = list(atomic_transaction(conn, tables_query).fetchall())
    assert len(tables) == 4
    tablenames = tuple(table[1] for table in tables)
    assert all(dataset.name not in table_name for table_name in tablenames)


# todo missing from runs table
# snapshot, completed timestamp, parameters (do we care), verify other metadata
# When should metadata be added. In the old dataset it used to be added as
# soon as you call add_metadata


# add a test to import from 0.26 data (missing parent dataset links)
