from qcodes.dataset.data_set import DataSet
from qcodes.dataset.sqlite_storage_interface import SqliteStorageInterface
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment, dataset)
import qcodes.dataset.sqlite_base as sqlite


def test_init_for_new_run(experiment):
    """
    Test that the initialisation of a brand new run works, i.e. that the
    database correctly gets new entries and also that the DataSet object
    has all attributes
    """

    conn = experiment.conn

    no_of_runs = len(sqlite.get_runs(conn, experiment.exp_id))
    assert no_of_runs == 0

    ds = DataSet(guid=None, conn=conn)
    assert isinstance(ds.dsi, SqliteStorageInterface)

    # check that the run got into the database

    no_of_runs = len(sqlite.get_runs(conn, experiment.exp_id))
    assert no_of_runs == 1

    # check that all traits "work"

    for trait in DataSet.persistent_traits:
        getattr(ds, trait)
