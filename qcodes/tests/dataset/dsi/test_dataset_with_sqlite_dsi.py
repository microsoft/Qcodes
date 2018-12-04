from pathlib import Path

import numpy as np
import pytest

from qcodes import ParamSpec
from qcodes.dataset.data_set import DataSet, CompletedError
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
    information is just stored inside dataset. Adding a parameter to a
    started or completed DataSet is an error.
    """
    conn = experiment.conn
    db_file = path_to_dbfile(conn)

    ds = DataSet(guid=None, storageinterface=SqliteStorageInterface, conn=conn)

    spec = ParamSpec('x', 'numeric')

    with raise_if_file_changed(db_file):
        ds.add_parameter(spec)

    expected_descr = RunDescriber(InterDependencies(spec))
    assert expected_descr == ds.description

    # Make DataSet started by adding a first result and try to add a parameter

    ds.add_result({'x': 1})

    with pytest.raises(RuntimeError, match='It is not allowed to add '
                                           'parameters to a started run'):
        ds.add_parameter(spec)

    # Mark DataSet as completed and try to add a parameter

    ds.mark_completed()

    with pytest.raises(RuntimeError, match='It is not allowed to add '
                                           'parameters to a started run'):
        ds.add_parameter(spec)


@pytest.mark.parametrize('first_add_using_add_result', (True, False))
def test_add_results(experiment, first_add_using_add_result, request):
    """
    Test adding results to the dataset. Assertions are made directly in the
    sqlite database.
    """
    conn = experiment.conn

    control_conn = sqlite.connect(experiment.path_to_db)
    request.addfinalizer(control_conn.close)

    # Initialize a dataset

    ds = DataSet(guid=None, conn=conn)

    # Assert initial state

    assert False is ds.completed

    # Add parameters to the dataset

    x_spec = ParamSpec('x', 'numeric')
    y_spec = ParamSpec('y', 'array')

    ds.add_parameter(x_spec)
    ds.add_parameter(y_spec)

    # Now let's add results

    expected_x = []
    expected_y = []

    # We need to test that both `add_result` and `add_results` (with 's')
    # perform "start actions" when they are called on a non-started run,
    # hence this parametrization
    if first_add_using_add_result:
        x_val = 25
        y_val = np.random.random_sample(3)
        expected_x.append([x_val])
        expected_y.append([y_val])
        ds.add_result({'x': x_val, 'y': y_val})
        len_after_first_add = 1
    else:
        x_val_1 = 42
        x_val_2 = 53
        y_val_1 = np.random.random_sample(3)
        y_val_2 = np.random.random_sample(3)
        expected_x.append([x_val_1])
        expected_x.append([x_val_2])
        expected_y.append([y_val_1])
        expected_y.append([y_val_2])
        len_before_add = ds.add_results([{'x': x_val_1, 'y': y_val_1},
                                         {'x': x_val_2, 'y': y_val_2}])
        assert len_before_add == 0
        len_after_first_add = 2

    # We also parametrize the second addition of results
    second_add_using_add_result = not first_add_using_add_result
    if second_add_using_add_result:
        x_val = 12
        y_val = np.random.random_sample(3)
        expected_x.append([x_val])
        expected_y.append([y_val])
        ds.add_result({'x': x_val, 'y': y_val})
    else:
        x_val_1 = 68
        x_val_2 = 75
        y_val_1 = np.random.random_sample(3)
        y_val_2 = np.random.random_sample(3)
        expected_x.append([x_val_1])
        expected_x.append([x_val_2])
        expected_y.append([y_val_1])
        expected_y.append([y_val_2])
        len_before_add = ds.add_results([{'x': x_val_1, 'y': y_val_1},
                                         {'x': x_val_2, 'y': y_val_2}])
        assert len_before_add == len_after_first_add

    # assert that data has been added to the database

    actual_x = sqlite.get_data(control_conn, ds.dsi.table_name, ['x'])
    assert actual_x == expected_x

    actual_y = sqlite.get_data(control_conn, ds.dsi.table_name, ['y'])
    np.testing.assert_allclose(actual_y, expected_y)

    # assert that we can't `add_result` and `add_results` to a completed dataset

    ds.mark_completed()

    with raise_if_file_changed(ds.dsi.path_to_db):

        with pytest.raises(CompletedError):
            ds.add_result({'x': 1})

        with pytest.raises(CompletedError):
            ds.add_results([{'x': 1}])


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

    ds.mark_completed()

    same_ds = DataSet(guid=guid, conn=control_conn)

    assert True is same_ds.completed
    assert True is same_ds.started

    # 4. Create a new dataset and mark it as completed without adding results

    ds = DataSet(guid=None, conn=conn)
    guid = ds.guid

    assert False is ds.completed
    assert False is ds.started

    ds.mark_completed()

    assert True is ds.completed
    assert True is ds.started

    same_ds = DataSet(guid=guid, conn=control_conn)

    assert True is same_ds.completed
    assert True is same_ds.started
