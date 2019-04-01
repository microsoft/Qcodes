from pathlib import Path

import numpy as np
import pytest

from qcodes import ParamSpec
from qcodes.dataset.data_set import DataSet, CompletedError
from qcodes.dataset.dependencies import InterDependencies
from qcodes.dataset.descriptions import RunDescriber
from qcodes.dataset.data_storage_interface import DataStorageInterface
from qcodes.dataset.sqlite_storage_interface import (SqliteReaderInterface,
                                                     SqliteWriterInterface)
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
    assert ds.dsi.reader.exp_id == experiment.exp_id

    assert ds.name == 'dataset'
    assert ds.dsi.reader.name == 'dataset'
    assert ds.dsi.reader.table_name == 'dataset-1-1'

    # check that all attributes are piped through correctly

    assert isinstance(ds.dsi, DataStorageInterface)
    # for a *new* run, the connection is given to the writer
    assert ds.dsi.writer.conn == conn
    assert ds.dsi.reader.conn != conn
    assert ds.dsi.reader.path_to_db == path_to_dbfile(conn)

    # check that the run got into the database

    no_of_runs = len(sqlite.get_runs(conn, experiment.exp_id))
    assert no_of_runs == 1

    # check that all traits "work"

    for trait in DataSet.persistent_traits:
        getattr(ds, trait)

    assert False is ds.completed
    assert False is ds._started


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
    assert ds.dsi.reader.exp_id == experiment.exp_id

    assert ds.name == ds_name
    assert ds.dsi.reader.name == ds_name
    assert ds.dsi.reader.table_name == f'{ds_name}-1-1'

    # check that all attributes are piped through correctly

    assert isinstance(ds.dsi.reader, SqliteReaderInterface)
    # for a *new* run, the connection is given to the writer
    assert ds.dsi.writer.conn == conn
    assert ds.dsi.reader.conn != conn
    assert ds.dsi.reader.path_to_db == path_to_dbfile(conn)

    # check that the run got into the database

    no_of_runs = len(sqlite.get_runs(conn, experiment.exp_id))
    assert no_of_runs == 1

    # check that all traits "work"

    for trait in DataSet.persistent_traits:
        getattr(ds, trait)

    assert False is ds.completed
    assert False is ds._started


def test_add_parameter(experiment):
    """
    Test adding new parameter to the dataset with sqlite storage interface.
    Adding a parameter does change the database, because storing parameter
    information also creates a column in the results table (if necessary).
    Adding a parameter to a started or completed DataSet is an error.
    """
    conn = experiment.conn
    db_file = path_to_dbfile(conn)

    ds = DataSet(guid=None, conn=conn)

    spec = ParamSpec('x', 'numeric')

    with pytest.raises(RuntimeError, match='File .* was modified'):
        with raise_if_file_changed(db_file):
            ds.add_parameter(spec)

    expected_descr = RunDescriber(InterDependencies(spec))
    assert expected_descr == ds.description

    # Make DataSet started and try to add a parameter

    ds.mark_started()

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

    ds.mark_started()

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

    actual_x = sqlite.get_data(control_conn, ds.dsi.writer.table_name, ['x'])
    assert actual_x == expected_x

    actual_y = sqlite.get_data(control_conn, ds.dsi.reader.table_name, ['y'])
    np.testing.assert_allclose(actual_y, expected_y)

    # assert that we can't `add_result` and `add_results`
    # to a completed dataset

    ds.mark_completed()

    with raise_if_file_changed(ds.dsi.reader.path_to_db):

        with pytest.raises(CompletedError):
            ds.add_result({'x': 1})

        with pytest.raises(CompletedError):
            ds.add_results([{'x': 1}])


@pytest.mark.parametrize('start_end', ((None, None),
                                       (2, None),
                                       (None, 1)))
def test_get_data(experiment, request, start_end):
    """
    Test get_data of DataSet, the data should come out as described in the
    method's docstring. Also cover 'start' and 'end' arguments with
    parametrized tests.
    """
    conn = experiment.conn

    control_conn = sqlite.connect(experiment.path_to_db)
    request.addfinalizer(control_conn.close)

    # Initialize a dataset with some results

    ds = DataSet(guid=None, conn=conn)

    x_spec = ParamSpec('x', 'numeric')
    y_spec = ParamSpec('y', 'array')
    ds.add_parameter(x_spec)
    ds.add_parameter(y_spec)
    ds.mark_started()

    expected_x = []
    expected_y = []
    expected_y_x_all = []

    x_val_1 = 42
    x_val_2 = 53
    y_val_1 = np.random.random_sample(3)
    y_val_2 = np.random.random_sample(3)
    expected_x.append([x_val_1])
    expected_x.append([x_val_2])
    expected_y.append([y_val_1])
    expected_y.append([y_val_2])
    expected_y_x_all.append([y_val_1, x_val_1])
    expected_y_x_all.append([y_val_2, x_val_2])
    ds.add_results([{'x': x_val_1, 'y': y_val_1},
                    {'x': x_val_2, 'y': y_val_2}])

    # assert that data has been added to the database

    actual_x = sqlite.get_data(control_conn, ds.dsi.reader.table_name, ['x'])
    assert actual_x == expected_x

    actual_y = sqlite.get_data(control_conn, ds.dsi.writer.table_name, ['y'])
    np.testing.assert_allclose(actual_y, expected_y)

    # Now get data using DataSet's get_data

    actual_x = ds.get_data('x')
    assert actual_x == expected_x

    actual_y = ds.get_data('y')
    np.testing.assert_allclose(actual_y, expected_y)

    # Here we test getting data for a given set of parameters, note the
    # reverse order of the parameters, as well as 'start' and 'end' arguments

    actual_y_x = ds.get_data('y', 'x', start=start_end[0], end=start_end[1])

    p_start = 0 if start_end[0] is None else start_end[0] - 1
    p_end = len(expected_y_x_all) if start_end[1] is None else start_end[1]
    expected_y_x = expected_y_x_all[p_start:p_end]

    assert len(actual_y_x) == len(expected_y_x)
    for act_y_x, exp_y_x in zip(actual_y_x, expected_y_x):
        assert len(act_y_x) == len(exp_y_x)
        for act_param, exp_param in zip(act_y_x, exp_y_x):
            np.testing.assert_allclose(act_param, exp_param)


def test_get_values(experiment, request):
    """
    Test get_values of DataSet, the data should come out as described in the
    method's docstring, and as the original implementation that used get_values
    function from sqlite_base.
    """
    conn = experiment.conn

    control_conn = sqlite.connect(experiment.path_to_db)
    request.addfinalizer(control_conn.close)

    # Initialize a dataset with some results

    ds = DataSet(guid=None, conn=conn)

    x_spec = ParamSpec('x', 'numeric')
    y_spec = ParamSpec('y', 'array')
    ds.add_parameter(x_spec)
    ds.add_parameter(y_spec)
    ds.mark_started()

    all_x = []
    all_y = []

    x_val_1 = 42
    x_val_2 = None
    y_val_1 = None
    y_val_2 = np.random.random_sample(3)
    all_x.append([x_val_1])
    all_x.append([x_val_2])
    all_y.append([y_val_1])
    all_y.append([y_val_2])
    ds.add_results([{'x': x_val_1, 'y': y_val_1},
                    {'x': x_val_2, 'y': y_val_2}])

    # assert that data has been added to the database

    actual_x = sqlite.get_data(control_conn, ds.dsi.reader.table_name, ['x'])
    assert actual_x == all_x

    actual_y = sqlite.get_data(control_conn, ds.dsi.reader.table_name, ['y'])
    assert len(actual_y) == len(all_y)
    for act_y, al_y in zip(actual_y, all_y):
        assert len(act_y) == len(al_y)
        for ac_y, a_y in zip(act_y, al_y):
            if isinstance(a_y, np.ndarray):
                np.testing.assert_allclose(ac_y, a_y)
            else:
                assert ac_y == a_y

    # Now get data using DataSet's get_values

    actual_x = ds.get_values('x')
    expected_x = [item for item in all_x
                  for subitem in item
                  if subitem is not None]
    assert actual_x == expected_x

    sqlite_actual_x = sqlite.get_values(control_conn,
                                        ds.dsi.reader.table_name, 'x')
    assert actual_x == sqlite_actual_x

    actual_y = ds.get_values('y')
    # `is not` removes a dimension in this particular case, hence extra `[]`
    # around the `all_y[...]` expression
    expected_y = [all_y[all_y is not None]]
    np.testing.assert_allclose(actual_y, expected_y)

    sqlite_actual_y = sqlite.get_values(control_conn,
                                        ds.dsi.writer.table_name, 'y')
    np.testing.assert_allclose(actual_y, sqlite_actual_y)


def test_get_setpoints(experiment, request):
    """
    Test get_setpoints of DataSet, the data should come out as described in the
    method's docstring, and as the original implementation that used
    get_setpoints function from sqlite_base.
    """
    conn = experiment.conn

    control_conn = sqlite.connect(experiment.path_to_db)
    request.addfinalizer(control_conn.close)

    # Initialize a dataset with some results

    ds = DataSet(guid=None, conn=conn)

    x_spec = ParamSpec('x', 'numeric')
    y_spec = ParamSpec('y', 'array')
    z_spec = ParamSpec('z', 'numeric', depends_on=[x_spec, y_spec])
    ds.add_parameter(x_spec)
    ds.add_parameter(y_spec)
    ds.add_parameter(z_spec)
    ds.mark_started()

    x_val_1 = 42
    x_val_2 = 43
    x_val_3 = None
    x_val_4 = 96
    y_val_1 = np.random.random_sample(3)
    y_val_2 = None
    y_val_3 = np.random.random_sample(3)
    y_val_4 = np.random.random_sample(3)
    z_val_1 = 666
    z_val_2 = 999
    z_val_3 = 777
    z_val_4 = None
    all_x = [[x_val_1], [x_val_2], [x_val_3], [x_val_4]]
    all_y = [[y_val_1], [y_val_2], [y_val_3], [y_val_4]]
    all_z = [[z_val_1], [z_val_2], [z_val_3], [z_val_4]]

    ds.add_results([{'x': x_val_1, 'y': y_val_1, 'z': z_val_1},
                    {'x': x_val_2, 'y': y_val_2, 'z': z_val_2},
                    {'x': x_val_3, 'y': y_val_3, 'z': z_val_3},
                    {'x': x_val_4, 'y': y_val_4, 'z': z_val_4}])

    # assert that data has been added to the database

    actual_x = sqlite.get_data(control_conn, ds.dsi.writer.table_name, ['x'])
    assert actual_x == all_x

    actual_y = sqlite.get_data(control_conn, ds.dsi.reader.table_name, ['y'])
    assert len(actual_y) == len(all_y)
    for act_y, al_y in zip(actual_y, all_y):
        assert len(act_y) == len(al_y)
        for ac_y, a_y in zip(act_y, al_y):
            if isinstance(a_y, np.ndarray):
                np.testing.assert_allclose(ac_y, a_y)
            else:
                assert ac_y == a_y

    actual_z = sqlite.get_data(control_conn, ds.dsi.reader.table_name, ['z'])
    assert actual_z == all_z

    # Now get data using DataSet's get_setpoints

    actual_x_y = ds.get_setpoints('z')

    assert list(actual_x_y.keys()) == ['x', 'y']

    expected_x = [item for item, item_z in zip(all_x, all_z)
                  for subitem in item_z
                  if subitem is not None]
    assert actual_x_y['x'] == expected_x

    expected_y = [item for item, item_z in zip(all_y, all_z)
                  for subitem in item_z
                  if subitem is not None]
    assert len(actual_x_y['y']) == len(expected_y)
    for act_y, exp_y in zip(actual_x_y['y'], expected_y):
        assert len(act_y) == len(exp_y)
        for a_y, e_y in zip(act_y, exp_y):
            if isinstance(a_y, np.ndarray):
                np.testing.assert_allclose(a_y, e_y)
            else:
                assert a_y == e_y

    # Finally, compare the actuals with the output of the
    # get_setpoints function of sqlite_base module that was used as the
    # original implementation

    sqlite_actual_x_y = sqlite.get_setpoints(
        control_conn, ds.dsi.writer.table_name, 'z')

    assert list(actual_x_y.keys()) == list(sqlite_actual_x_y.keys())

    assert actual_x_y['x'] == sqlite_actual_x_y['x']

    assert len(actual_x_y['y']) == len(sqlite_actual_x_y['y'])
    for act_y, sql_y in zip(actual_x_y['y'], sqlite_actual_x_y['y']):
        assert len(act_y) == len(sql_y)
        for a_y, s_y in zip(act_y, sql_y):
            if isinstance(a_y, np.ndarray):
                np.testing.assert_allclose(a_y, s_y)
            else:
                assert a_y == s_y


def test_dataset_state_in_different_cases(experiment):
    """
    Test that DataSet's `_started`, `completed`, `pristine`, `running`
    properties are correct in various cases
    """
    conn = experiment.conn

    control_conn = sqlite.connect(experiment.path_to_db)

    # 1. Create a new dataset

    ds = DataSet(guid=None, conn=conn)
    guid = ds.guid

    assert False is ds.completed
    assert False is ds._started
    assert True is ds.pristine
    assert False is ds.running

    same_ds = DataSet(guid=guid, conn=control_conn)

    assert False is same_ds.completed
    assert False is same_ds._started
    assert True is same_ds.pristine
    assert False is same_ds.running

    # 2. Start this dataset

    ds.add_parameter(ParamSpec('x', 'numeric'))
    ds.mark_started()
    ds.add_result({'x': 0})

    assert True is ds._started
    assert False is ds.completed
    assert False is ds.pristine
    assert True is ds.running

    same_ds = DataSet(guid=guid, conn=control_conn)

    assert False is same_ds.completed
    assert True is same_ds._started
    assert False is same_ds.pristine
    assert True is same_ds.running

    # 3. Complete this dataset

    ds.mark_completed()

    same_ds = DataSet(guid=guid, conn=control_conn)

    assert True is same_ds.completed
    assert True is same_ds._started
    assert False is same_ds.pristine
    assert False is same_ds.running

    # 4. Create a new dataset and mark it as completed without adding results

    ds = DataSet(guid=None, conn=conn)
    guid = ds.guid
    ds.mark_started()

    assert False is ds.completed
    assert False is ds._started
    assert True is ds.pristine
    assert False is ds.running

    ds.mark_completed()

    assert True is ds.completed
    assert True is ds._started
    assert False is ds.pristine
    assert False is ds.running

    same_ds = DataSet(guid=guid, conn=control_conn)

    assert True is same_ds.completed
    assert True is same_ds._started
    assert False is same_ds.pristine
    assert False is same_ds.running
