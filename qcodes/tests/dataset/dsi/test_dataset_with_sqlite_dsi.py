from pathlib import Path

import pytest

from qcodes.dataset.data_set import DataSet
from qcodes.dataset.sqlite_storage_interface import SqliteStorageInterface
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment, dataset)
import qcodes.dataset.sqlite_base as sqlite
from qcodes.tests.dataset.temporary_databases import two_empty_temp_db_connections
from qcodes.dataset.database import path_to_dbfile
from qcodes.tests.dataset.test_database_extract_runs import raise_if_file_changed


def test_init_for_new_run(experiment):
    """
    Test that the initialisation of a brand new run works, i.e. that the
    database correctly gets new entries and also that the DataSet object
    has all attributes
    """

    conn = experiment.conn

    no_of_runs = len(sqlite.get_runs(conn, experiment.exp_id))
    assert no_of_runs == 0

    # check that this modifies the relevant file

    with pytest.raises(RuntimeError):
        with raise_if_file_changed(path_to_dbfile(conn)):
            ds = DataSet(guid=None, conn=conn)

    # check sqlite-database-related properties of sqlite dsi are correctly
    # assigned

    assert ds.exp_id == experiment.exp_id
    assert ds.dsi.exp_id == experiment.exp_id

    assert ds.name == 'dataset'
    assert ds.dsi.name == 'dataset'
    assert ds.dsi.table_name == 'dataset-1-1'

    # check that all attributes are piped through correctly

    assert isinstance(ds.dsi, SqliteStorageInterface)
    assert ds.dsi.conn == conn
    assert ds.dsi.path_to_db == path_to_dbfile(conn)

    # check that the run got into the database

    no_of_runs = len(sqlite.get_runs(conn, experiment.exp_id))
    assert no_of_runs == 1

    # check that all traits "work"

    for trait in DataSet.persistent_traits:
        getattr(ds, trait)


def test_init_for_new_run_with_given_experiment_and_name(experiment):
    """
    Test that the initialisation of a new run within a given experiment and
    given name works
    """
    conn = experiment.conn
    ds_name = 'extraordinary-name'

    no_of_runs = len(sqlite.get_runs(conn, experiment.exp_id))
    assert no_of_runs == 0

    # check that this modifies the relevant file

    with pytest.raises(RuntimeError):
        with raise_if_file_changed(path_to_dbfile(conn)):
            ds = DataSet(guid=None, conn=conn,
                         exp_id=experiment.exp_id,
                         name=ds_name)

    # check sqlite-database-related properties of sqlite dsi are correctly
    # assigned

    assert ds.exp_id == experiment.exp_id
    assert ds.dsi.exp_id == experiment.exp_id

    assert ds.name == ds_name
    assert ds.dsi.name == ds_name
    assert ds.dsi.table_name == f'{ds_name}-1-1'

    # check that all attributes are piped through correctly

    assert isinstance(ds.dsi, SqliteStorageInterface)
    assert ds.dsi.conn == conn
    assert ds.dsi.path_to_db == path_to_dbfile(conn)

    # check that the run got into the database

    no_of_runs = len(sqlite.get_runs(conn, experiment.exp_id))
    assert no_of_runs == 1

    # check that all traits "work"

    for trait in DataSet.persistent_traits:
        getattr(ds, trait)
