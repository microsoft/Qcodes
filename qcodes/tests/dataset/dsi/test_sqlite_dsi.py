import re
import time
import json

import numpy as np
import pytest

from qcodes import ParamSpec, load_experiment_by_name
from qcodes.dataset.database import get_DB_location
from qcodes.dataset.data_storage_interface import MetaData
from qcodes.dataset.dependencies import InterDependencies
from qcodes.dataset.descriptions import RunDescriber
from qcodes.dataset.guids import generate_guid
from qcodes.dataset.sqlite_base import (connect,
                                        create_run,
                                        get_data,
                                        get_experiments,
                                        get_metadata,
                                        get_run_counter,
                                        get_runs,
                                        RUNS_TABLE_COLUMNS)

from qcodes.dataset.sqlite_storage_interface import (SqliteReaderInterface,
                                                     SqliteWriterInterface)
from qcodes.dataset.data_storage_interface import (DataStorageInterface,
                                                   MetaData)
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment, dataset)
from qcodes.tests.dataset.test_database_creation_and_upgrading import \
    error_caused_by
from qcodes.dataset.database import path_to_dbfile

# IMPORTANT: use pytest.xfail at the edn of a test function to mark tests
# that are passing but should be improved, and write in the 'reason' what
# there is to be improved.
from qcodes.utils.helpers import NumpyJSONEncoder


def test_init_no_guid():
    """Test that dsi requires guid as an argument"""
    match_str = re.escape("__init__() missing 1 required"
                          " positional argument: 'guid'")
    with pytest.raises(TypeError, match=match_str):
        SqliteReaderInterface()
    with pytest.raises(TypeError, match=match_str):
        SqliteWriterInterface()


def test_init_and_create_new_run(experiment):
    """
    Test initialising with Sqlite for a new run. The few steps taken in the
    initialisation procedure mimick those performed by the DataSet
    """
    conn = experiment.conn
    guid = generate_guid()

    check_time = time.time()  # used below as a sanity check for creation time
    time.sleep(0.001)

    # ensure there are no runs in the database
    assert [] == get_runs(conn)

    dsi_reader = SqliteReaderInterface(guid, conn=conn)

    assert experiment.conn is dsi_reader.conn
    assert guid == dsi_reader.guid
    assert dsi_reader.run_id is None
    assert experiment.path_to_db == dsi_reader.path_to_db
    assert not(dsi_reader.run_exists())
    assert dsi_reader.exp_id is None
    assert dsi_reader.name is None
    assert dsi_reader.table_name is None
    assert dsi_reader.counter is None

    dsi_writer = SqliteWriterInterface(guid, conn=conn)

    # That was the bare __init__. Now create the run
    dsi_writer.create_run()

    assert dsi_reader.run_exists()
    assert dsi_writer.run_id == 1
    runs_rows = get_runs(conn)
    assert 1 == len(runs_rows)
    assert 1 == runs_rows[0]['run_id']
    assert experiment.exp_id == dsi_writer.exp_id
    assert runs_rows[0]['name'] == dsi_writer.name
    assert runs_rows[0]['result_table_name'] == dsi_writer.table_name
    assert runs_rows[0]['result_counter'] == dsi_writer.counter

    md = dsi_reader.retrieve_meta_data()

    assert md.run_completed is None
    empty_desc = RunDescriber(InterDependencies())
    assert empty_desc.to_json() == md.run_description
    assert md.run_started is None
    assert md.snapshot is None
    assert {} == md.tags
    assert 1 == md.tier
    assert 'dataset' == md.name
    assert experiment.name == md.exp_name
    assert experiment.sample_name == md.sample_name
    assert experiment.exp_id == dsi_reader.exp_id
    assert md.name == dsi_reader.name
    assert 'dataset-1-1' == dsi_reader.table_name
    assert 1 == dsi_reader.counter

    # Test builtin convertion to dict works
    md_dict = md.asdict()
    assert isinstance(md_dict, dict)
    assert md_dict['run_completed'] is None
    empty_desc = RunDescriber(InterDependencies())
    assert empty_desc.to_json() == md_dict['run_description']
    assert md_dict['run_started'] is None
    assert md_dict['snapshot'] is None
    assert {} == md_dict['tags']
    assert 1 == md_dict['tier']
    assert 'dataset' == md_dict['name']
    assert experiment.name == md_dict['exp_name']
    assert experiment.sample_name == md_dict['sample_name']


def test_create_run_for_given_experiment_name_and_sample_name(empty_temp_db,
                                                              request):
    conn = connect(get_DB_location())
    request.addfinalizer(conn.close)

    guid = generate_guid()

    exp_name = 'new_exp'
    sample_name = 'good_sample'

    dsi_reader = SqliteReaderInterface(guid, conn=conn)

    assert not (dsi_reader.run_exists())

    dsi_writer = SqliteWriterInterface(guid, conn=conn)

    # That was the bare __init__. Now create the run
    dsi_writer.create_run(exp_name=exp_name, sample_name=sample_name)

    # Try to load experiment to see if it has been created
    experiment = load_experiment_by_name(exp_name, sample_name, conn)
    assert experiment.name == exp_name
    assert experiment.sample_name == sample_name

    assert experiment.exp_id == dsi_writer.exp_id

    assert dsi_reader.run_exists()
    assert dsi_writer.run_id == 1
    runs_rows = get_runs(conn)
    assert 1 == len(runs_rows)
    assert 1 == runs_rows[0]['run_id']
    assert runs_rows[0]['name'] == dsi_writer.name
    assert runs_rows[0]['result_table_name'] == dsi_writer.table_name
    assert runs_rows[0]['result_counter'] == dsi_writer.counter

    md = dsi_reader.retrieve_meta_data()

    assert experiment.name == md.exp_name
    assert experiment.sample_name == md.sample_name
    assert experiment.exp_id == dsi_reader.exp_id

    assert md.run_completed is None
    empty_desc = RunDescriber(InterDependencies())
    assert empty_desc.to_json() == md.run_description
    assert md.run_started is None
    assert md.snapshot is None
    assert {} == md.tags
    assert 1 == md.tier
    assert 'dataset' == md.name
    assert md.name == dsi_reader.name
    assert 'dataset-1-1' == dsi_reader.table_name
    assert 1 == dsi_reader.counter


def test_create_run_for_given_and_existing_experiment_name_and_sample_name(
        experiment):
    conn = experiment.conn
    guid = generate_guid()

    dsi_reader = SqliteReaderInterface(guid, conn=conn)

    assert not (dsi_reader.run_exists())

    dsi_writer = SqliteWriterInterface(guid, conn=conn)

    # That was the bare __init__. Now create the run
    dsi_writer.create_run(exp_name=experiment.name,
                          sample_name=experiment.sample_name)

    assert experiment.exp_id == dsi_writer.exp_id

    assert dsi_reader.run_exists()
    assert dsi_writer.run_id == 1
    runs_rows = get_runs(conn)
    assert 1 == len(runs_rows)
    assert 1 == runs_rows[0]['run_id']
    assert runs_rows[0]['name'] == dsi_writer.name
    assert runs_rows[0]['result_table_name'] == dsi_writer.table_name
    assert runs_rows[0]['result_counter'] == dsi_writer.counter

    md = dsi_reader.retrieve_meta_data()

    assert experiment.name == md.exp_name
    assert experiment.sample_name == md.sample_name
    assert experiment.exp_id == dsi_reader.exp_id

    assert md.run_completed is None
    empty_desc = RunDescriber(InterDependencies())
    assert empty_desc.to_json() == md.run_description
    assert md.run_started is None
    assert md.snapshot is None
    assert {} == md.tags
    assert 1 == md.tier
    assert 'dataset' == md.name
    assert md.name == dsi_reader.name
    assert 'dataset-1-1' == dsi_reader.table_name
    assert 1 == dsi_reader.counter


def test_init__load_existing_run(experiment):
    """Test initialising dsi for an existing run"""
    conn = experiment.conn
    guid = generate_guid()
    name = "existing-dataset"
    _, run_id, __ = create_run(conn, experiment.exp_id, name, guid)

    dsi = SqliteReaderInterface(guid, conn=conn)

    assert experiment.conn is dsi.conn
    assert guid == dsi.guid
    assert run_id is 1
    assert experiment.path_to_db == dsi.path_to_db
    assert dsi.run_id is None
    assert None is dsi.exp_id
    assert None is dsi.name
    assert None is dsi.table_name
    assert None is dsi.counter

    # that was the bare init, now load the run

    md = dsi.retrieve_meta_data()

    assert dsi.run_id == run_id
    assert None is md.run_completed
    assert RunDescriber(InterDependencies()).to_json() == md.run_description
    assert md.run_started is None
    assert None is md.snapshot
    assert {} == md.tags
    assert 1 == md.tier
    assert name == md.name
    assert experiment.name == md.exp_name
    assert experiment.sample_name == md.sample_name
    assert experiment.exp_id == dsi.exp_id
    assert name == dsi.name
    assert name+'-1-1' == dsi.table_name
    assert 1 == dsi.counter


def test_retrieve_metadata_empty_run(experiment):
    """Test dsi.retrieve_metadata for an empty run"""
    t_before_run_init = time.perf_counter()

    guid = generate_guid()
    conn = experiment.conn
    dsi = SqliteReaderInterface(guid, conn=conn)
    dsi_writer = SqliteWriterInterface(guid, conn=conn)

    with pytest.raises(ValueError):
        dsi.retrieve_meta_data()

    dsi_writer.create_run()

    md = dsi.retrieve_meta_data()

    assert md is not None
    assert isinstance(md, MetaData)
    assert None is md.run_completed
    assert RunDescriber(InterDependencies()).to_json() == md.run_description
    assert md.run_started is None
    assert None is md.snapshot
    assert {} == md.tags
    assert 1 == md.tier
    assert 'dataset' == md.name
    assert experiment.name == md.exp_name
    assert experiment.sample_name == md.sample_name


def test_store_results(experiment, request):
    """
    Test storing results via sqlite dsi. Also test
    retrieve_number_of_results method along the way
    """
    guid = generate_guid()
    conn = experiment.conn
    dsi_writer = SqliteWriterInterface(guid, conn=conn)

    # we use a different connection in order to make sure that the
    # transactions get committed and the database file gets indeed changed to
    # contain the data points; for the same reason we use another dsi instance
    control_conn = connect(experiment.path_to_db)
    request.addfinalizer(control_conn.close)
    control_dsi = SqliteReaderInterface(guid, conn=control_conn)

    with pytest.raises(RuntimeError,
                       match='Rolling back due to unhandled exception') as e:
        control_dsi.retrieve_number_of_results()
    assert error_caused_by(e, 'Expected one row')

    dsi_writer.create_run()

    assert 0 == control_dsi.retrieve_number_of_results()

    specs = [ParamSpec("x", "numeric"), ParamSpec("y", "array")]
    desc = RunDescriber(InterDependencies(*specs))

    # Add specs for parameters via metadata
    dsi_writer.store_meta_data(MetaData(run_description=desc.to_json()))

    dsi_writer.prepare_for_storing_results()

    assert 0 == control_dsi.retrieve_number_of_results()

    expected_x = []
    expected_y = []

    # store_results where the results dict has single value per parameter
    n_res_1 = 10
    for x in range(n_res_1):
        y = np.random.random_sample(10)
        xx = [x]
        yy = [y]
        expected_x.append(xx)
        expected_y.append(yy)

        dsi_writer.store_results({"x": xx, "y": yy})

        n_res = x + 1
        assert n_res == control_dsi.retrieve_number_of_results()

    # store_results where the results dict has multiple values per parameter
    n_pts = 3
    n_res_2 = 3
    for x in range(n_res_2):
        y = np.random.random_sample(10)
        xx = [x] * n_pts
        yy = [y] * n_pts
        for xx_ in xx:
            expected_x.append([xx_])
        for yy_ in yy:
            expected_y.append([yy_])

        dsi_writer.store_results({"x": xx, "y": yy})

        n_res = n_res_1 + (x + 1) * n_pts
        assert n_res == control_dsi.retrieve_number_of_results()

    actual_x = get_data(control_conn, dsi_writer.table_name, ['x'])
    assert actual_x == expected_x

    actual_y = get_data(control_conn, dsi_writer.table_name, ['y'])
    np.testing.assert_allclose(actual_y, expected_y)


def test_replay_results(experiment, request):
    """
    Test retrieving results via sqlite dsi.
    """
    guid = generate_guid()
    conn = experiment.conn
    reader_conn = connect(path_to_dbfile(conn))
    dsi = DataStorageInterface(guid,
                               reader=SqliteReaderInterface,
                               writer=SqliteWriterInterface,
                               writer_kwargs={'conn': conn},
                               reader_kwargs={'conn': reader_conn})

    # we use a different connection in order to make sure that the
    # transactions get committed and the database file gets indeed changed to
    # contain the data points; for the same reason we use another dsi instance
    control_conn = connect(experiment.path_to_db)
    request.addfinalizer(control_conn.close)

    # 1. test replaying if the run does not exist

    match_str = re.escape(f"No run with guid {guid} exists.")
    with pytest.raises(ValueError, match=match_str):
        dsi.replay_results()

    # create the run

    dsi.create_run()
    dsi.retrieve_meta_data()  # needed to "prepare" the reader

    # test replaying from an empty run

    results_iterator = dsi.replay_results()
    assert 0 == len(results_iterator)
    assert [] == list(results_iterator)

    # add parameters and prepare for storing data

    specs = [ParamSpec("x", "numeric"), ParamSpec("y", "array")]
    desc = RunDescriber(InterDependencies(*specs))
    dsi.store_meta_data(MetaData(run_description=desc.to_json()))

    dsi.prepare_for_storing_results()

    # test replaying from "prepared run"

    results_iterator = dsi.replay_results()
    assert 0 == len(results_iterator)
    assert [] == list(results_iterator)

    # add some results

    expected_x = []
    expected_y = []
    expected_results = []

    n_res = 8
    for x in range(n_res):
        y = np.random.random_sample(10)
        xx = [x]
        yy = [y]
        expected_x.append(xx)
        expected_y.append(yy)

        result = {"x": xx, "y": yy}
        expected_results.append(result)

        dsi.store_results(result)

    # ensure that results were physically added

    actual_x = get_data(control_conn, dsi.reader.table_name, ['x'])
    assert actual_x == expected_x

    actual_y = get_data(control_conn, dsi.writer.table_name, ['y'])
    np.testing.assert_allclose(actual_y, expected_y)

    # test replaying all stored results

    results_iterator = dsi.replay_results()

    assert n_res == len(results_iterator)

    actual_results = list(results_iterator)
    for act, exp in zip(actual_results, expected_results):
        assert list(exp.keys()) == list(act.keys())
        for act_item, exp_item in zip(act.values(), exp.values()):
            np.testing.assert_allclose(act_item, exp_item)

    # test replaying some of the stored results

    results_iterator = dsi.replay_results(start=2, stop=4)

    assert 3 == len(results_iterator)

    actual_results = list(results_iterator)
    for act, exp in zip(actual_results, expected_results[2-1:4]):
        assert list(exp.keys()) == list(act.keys())
        for act_item, exp_item in zip(act.values(), exp.values()):
            np.testing.assert_allclose(act_item, exp_item)


def test_store_meta_data__run_completed(experiment):
    guid = generate_guid()
    conn = experiment.conn
    reader_conn = connect(path_to_dbfile(conn))
    dsi = DataStorageInterface(guid,
                               reader=SqliteReaderInterface,
                               writer=SqliteWriterInterface,
                               writer_kwargs={'conn': conn},
                               reader_kwargs={'conn': reader_conn})
    dsi.create_run()

    control_conn = connect(experiment.path_to_db)

    # assert initial state

    check = "SELECT completed_timestamp,is_completed FROM runs WHERE run_id = ?"
    cursor = control_conn.execute(check, (dsi.writer.run_id,))
    row = cursor.fetchall()[0]
    assert 0 == row['is_completed']
    assert None is row['completed_timestamp']

    # store metadata

    some_time = time.time()
    dsi.store_meta_data(MetaData(run_completed=some_time))

    # assert metadata was successfully stored

    cursor = control_conn.execute(check, (dsi.writer.run_id,))
    row = cursor.fetchall()[0]
    assert 1 == row['is_completed']
    assert np.allclose(some_time, row['completed_timestamp'])


def test_store_meta_data__run_description(experiment):
    guid = generate_guid()
    conn = experiment.conn
    reader_conn = connect(path_to_dbfile(conn))
    dsi = DataStorageInterface(guid,
                               reader=SqliteReaderInterface,
                               writer=SqliteWriterInterface,
                               writer_kwargs={'conn': conn},
                               reader_kwargs={'conn': reader_conn})
    dsi.create_run()

    control_conn = connect(experiment.path_to_db)

    # assert initial state

    empty_desc = RunDescriber(InterDependencies())
    empty_desc_json = empty_desc.to_json()

    check = "SELECT run_description FROM runs WHERE run_id = ?"
    cursor = control_conn.execute(check, (dsi.writer.run_id,))
    row = cursor.fetchall()[0]
    assert empty_desc_json == row['run_description']

    # store metadata

    some_desc = RunDescriber(InterDependencies(ParamSpec('x', 'array')))
    some_desc_json = some_desc.to_json()
    dsi.store_meta_data(MetaData(run_description=some_desc.to_json()))

    # assert metadata was successfully stored

    cursor = control_conn.execute(check, (dsi.writer.run_id,))
    row = cursor.fetchall()[0]
    assert some_desc_json == row['run_description']


def test_store_meta_data__tags(experiment):
    guid = generate_guid()
    conn = experiment.conn
    reader_conn = connect(path_to_dbfile(conn))
    dsi = DataStorageInterface(guid,
                               reader=SqliteReaderInterface,
                               writer=SqliteWriterInterface,
                               writer_kwargs={'conn': conn},
                               reader_kwargs={'conn': reader_conn})
    dsi.create_run()

    control_conn = connect(experiment.path_to_db)

    # assert initial state

    sql = "SELECT * FROM runs WHERE run_id = ?"
    cursor = conn.execute(sql, (dsi.writer.run_id,))
    sql_result = dict(cursor.fetchall()[0])
    standard_columns = set(RUNS_TABLE_COLUMNS)
    actual_columns = set(sql_result.keys())
    assert standard_columns == actual_columns

    # 1. Store metadata

    tags_1 = {'run_is_good': False}

    dsi.store_meta_data(MetaData(tags=tags_1))

    # assert metadata was successfully stored

    cursor = control_conn.execute(sql, (dsi.writer.run_id,))
    sql_result = dict(cursor.fetchall()[0])
    expected_columns = {*standard_columns, 'run_is_good'}
    actual_columns = set(sql_result.keys())
    assert expected_columns == actual_columns

    # assert what we can retrieve

    md = dsi.retrieve_meta_data()
    assert md.tags == tags_1

    # 2. Store more metadata

    tags_2 = {**tags_1, 'evil_tag': 'not_really'}

    dsi.store_meta_data(MetaData(tags=tags_2))

    # assert metadata was successfully stored

    cursor = control_conn.execute(sql, (dsi.writer.run_id,))
    sql_result = dict(cursor.fetchall()[0])
    expected_columns = {*standard_columns, 'run_is_good', 'evil_tag'}
    actual_columns = set(sql_result.keys())
    assert expected_columns == actual_columns

    # assert what we can retrieve

    md = dsi.reader.retrieve_meta_data()
    assert md.tags == tags_2

    # 3. Store a different metadata

    # NOTE that in the current implementation, it is not possible to remove
    # already added metadata.

    tags_3 = {'very_different': 123.4}

    dsi.store_meta_data(MetaData(tags=tags_3))

    # assert metadata was successfully stored

    cursor = control_conn.execute(sql, (dsi.reader.run_id,))
    sql_result = dict(cursor.fetchall()[0])
    expected_columns = {*standard_columns,
                        'run_is_good', 'evil_tag', 'very_different'}
    actual_columns = set(sql_result.keys())
    assert expected_columns == actual_columns

    # assert what we can retrieve

    md = dsi.retrieve_meta_data()
    tags_all = {**tags_3, **tags_2}
    assert md.tags == tags_all


def test_store_meta_data__snapshot(experiment, request):
    guid = generate_guid()
    conn = experiment.conn
    reader_conn = connect(path_to_dbfile(conn))
    request.addfinalizer(reader_conn.close)
    dsi = DataStorageInterface(guid,
                               reader=SqliteReaderInterface,
                               writer=SqliteWriterInterface,
                               writer_kwargs={'conn': conn},
                               reader_kwargs={'conn': reader_conn})
    dsi.create_run()

    control_conn = connect(experiment.path_to_db)
    request.addfinalizer(control_conn.close)

    # assert initial state

    with pytest.raises(RuntimeError,
                       match='Rolling back due to unhandled exception') as e:
        get_metadata(dsi.writer.conn, 'snapshot', dsi.writer.table_name)
    assert error_caused_by(e, 'no such column: snapshot')

    md = dsi.retrieve_meta_data()
    assert None is md.snapshot

    # Store snapshot

    snap_1 = {'station': 'Q'}

    dsi.store_meta_data(MetaData(snapshot=snap_1))

    # assert snapshot was successfully stored

    snap_1_json = json.dumps(snap_1, cls=NumpyJSONEncoder)
    expected_snapshot = get_metadata(control_conn, 'snapshot',
                                     dsi.writer.table_name)
    assert expected_snapshot == snap_1_json

    # assert what we can retrieve

    md = dsi.retrieve_meta_data()
    assert md.snapshot == snap_1

    dsi.reader.conn.close()


def test_create_run_in_empty_db(empty_temp_db, request):
    guid = generate_guid()
    writer_conn = connect(get_DB_location())
    reader_conn = connect(get_DB_location())
    request.addfinalizer(reader_conn.close)
    request.addfinalizer(writer_conn.close)
    dsi = DataStorageInterface(guid,
                               reader=SqliteReaderInterface,
                               writer=SqliteWriterInterface,
                               writer_kwargs={'conn': writer_conn},
                               reader_kwargs={'conn': reader_conn})
    assert len(get_experiments(reader_conn)) == 0

    # First all the ways create_run will NOT create an experiment in an
    # empty DB

    match = re.escape("No experiments found. "
                      "You can start a new one with:"
                      " new_experiment(name, sample_name)")
    with pytest.raises(ValueError, match=match):
        dsi.create_run()

    with pytest.raises(RuntimeError):
        dsi.create_run(exp_id=1)

    exp_name = None
    sample_name = 'some_sample'
    match = re.escape(f'Got values for exp_name: {exp_name} and '
                      f'sample_name: {sample_name}. They must both '
                      'be None or both be not-None.')
    with pytest.raises(ValueError, match=match):
        dsi.create_run(exp_name=exp_name, sample_name=sample_name)

    # Then finally create an experiment
    dsi.create_run(exp_name='my_experiment', sample_name='best_sample_ever')

    # note: by using the reader_conn here, we also assert that all changes
    # were committed
    assert len(get_experiments(reader_conn)) == 1

    # now check that we can address that experiment with exp_id
    dsi.create_run(exp_id=1)

    assert get_run_counter(reader_conn, exp_id=1) == 2
