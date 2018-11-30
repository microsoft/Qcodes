from qcodes.dataset.data_set import DataSet
from qcodes.dataset.sqlite_storage_interface import SqliteStorageInterface
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment, dataset)


def test_init_for_new_run(experiment):
    ds = DataSet(run_id=None)

    assert isinstance(ds.data_storage_interface, SqliteStorageInterface)
