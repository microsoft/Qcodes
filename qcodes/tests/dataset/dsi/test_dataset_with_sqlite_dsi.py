from pathlib import Path

import pytest

from qcodes import ParamSpec
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.dependencies import InterDependencies
from qcodes.dataset.descriptions import RunDescriber
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

    assert False is ds.completed
    assert False is ds.started


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

    assert False is ds.completed
    assert False is ds.started


def test_add_parameter(experiment):
    """
    Test adding new parameter to the dataset with sqlite storage interface.
    Adding a parameter does not do anything with the database, parameter
    information is just stored inside dataset.
    """
    conn = experiment.conn
    db_file = path_to_dbfile(conn)

    ds = DataSet(guid=None, storageinterface=SqliteStorageInterface, conn=conn)

    spec = ParamSpec('x', 'array')

    with raise_if_file_changed(db_file):
        ds.add_parameter(spec)

    expected_descr = RunDescriber(InterDependencies(spec))
    assert expected_descr == ds.description


def test_run_is_started_in_different_cases(experiment):
    """Test that DataSet's `started` property is correct in various cases"""
    conn = experiment.conn

    control_conn = sqlite.connect(experiment.path_to_db)

    # 1. Create a new dataset

    ds = DataSet(guid=None, conn=conn)
    guid = ds.guid

    assert False is ds.completed
    assert False is ds.started

    same_ds = DataSet(guid=guid, conn=control_conn)

    assert False is same_ds.completed
    assert False is same_ds.started

    # 2. Start this dataset

    ds.add_parameter(ParamSpec('x', 'numeric'))
    ds.add_result({'x': 0})

    assert True is ds.started
    assert False is ds.completed

    same_ds = DataSet(guid=guid, conn=control_conn)

    assert False is same_ds.completed
    assert True is same_ds.started

    # 3. Complete this dataset

    ds.mark_complete()

    same_ds = DataSet(guid=guid, conn=control_conn)

    assert True is same_ds.completed
    assert True is same_ds.started

    # 4. Create a new dataset and mark it as completed without adding results

    # Note that when a dataset is new and it has been marked completed
    # without ever adding any results, the main instance will remain with
    # started==False, while a re-loaded instance will have started==True.

    ds = DataSet(guid=None, conn=conn)
    guid = ds.guid

    assert False is ds.completed
    assert False is ds.started

    ds.mark_complete()

    assert True is ds.completed
    assert False is ds.started  # <<== NOTE the difference !!!

    same_ds = DataSet(guid=guid, conn=control_conn)

    assert True is same_ds.completed
    assert True is same_ds.started  # <<== NOTE the difference !!!
