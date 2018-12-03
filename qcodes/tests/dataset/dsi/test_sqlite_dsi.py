import re
import time

import numpy as np
import pytest

from qcodes.dataset.data_storage_interface import MetaData
from qcodes.dataset.dependencies import InterDependencies
from qcodes.dataset.descriptions import RunDescriber
from qcodes.dataset.guids import generate_guid
from qcodes.dataset.sqlite_base import create_run, get_runs, connect, get_data
from qcodes.dataset.sqlite_storage_interface import SqliteStorageInterface
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment, dataset)


# IMPORTANT: use pytest.xfail at the edn of a test function to mark tests
# that are passing but should be improved, and write in the 'reason' what
# there is to be improved.


def test_init_no_guid():
    """Test that dsi requires guid as an argument"""
    match_str = re.escape("__init__() missing 1 required"
                          " positional argument: 'guid'")
    with pytest.raises(TypeError, match=match_str):
        SqliteStorageInterface()


def test_create_dataset_pass_both_connection_and_path_to_db(experiment):
    with pytest.raises(ValueError, match="Both `path_to_db` and `conn` "
                                         "arguments have been passed together "
                                         "with non-None values. This is not "
                                         "allowed."):
        some_valid_connection = experiment.conn
        _ = SqliteStorageInterface('some valid guid',
                                   path_to_db="some valid path",
                                   conn=some_valid_connection)


def test_init_and_create_new_run(experiment):
    """
    Test initialising dsi for a new run. The few steps taken in the
    initialisation procedure mimick those performed by the DataSet
    """
    conn = experiment.conn
    guid = generate_guid()

    check_time = time.time()  # used below as a sanity check for creation time
    time.sleep(0.001)

    # ensure there are no runs in the database
    assert [] == get_runs(conn)

    dsi = SqliteStorageInterface(guid, conn=conn)

    assert experiment.conn is dsi.conn
    assert guid == dsi.guid
    assert dsi.run_id is None
    assert experiment.path_to_db == dsi.path_to_db
    assert not(dsi.run_exists())

    # That was the bare __init__. Now create the run
    dsi.create_run()

    assert dsi.run_exists()
    assert dsi.run_id == 1
    runs_rows = get_runs(conn)
    assert 1 == len(runs_rows)
    assert 1 == runs_rows[0]['run_id']

    md = dsi.retrieve_meta_data()

    assert md.run_completed is None
    assert RunDescriber(InterDependencies()) == md.run_description
    assert md.run_started > check_time
    assert md.snapshot is None
    assert {} == md.tags
    assert 1 == md.tier
    assert 'dataset' == md.name
    assert experiment.name == md.exp_name
    assert experiment.sample_name == md.sample_name

    pytest.xfail('more assertions on the fact that run is created are needed')


def test_init__load_existing_run(experiment):
    """Test initialising dsi for an existing run"""
    conn = experiment.conn
    guid = generate_guid()
    name = "existing-dataset"
    _, run_id, __ = create_run(conn, experiment.exp_id, name, guid)

    dsi = SqliteStorageInterface(guid, conn=conn)

    assert experiment.conn is dsi.conn
    assert guid == dsi.guid
    assert run_id is 1
    assert experiment.path_to_db == dsi.path_to_db
    assert dsi.run_id is None

    # that was the bare init, now load the run

    md = dsi.retrieve_meta_data()

    assert dsi.run_id == run_id
    assert None is md.run_completed
    assert RunDescriber(InterDependencies()) == md.run_description
    assert md.run_started > 0
    assert None is md.snapshot
    assert {} == md.tags
    assert 1 == md.tier
    assert name == md.name
    assert experiment.name == md.exp_name
    assert experiment.sample_name == md.sample_name

    pytest.xfail(
        'more assertions needed for the fact that we loaded existing runs')


def test_retrieve_metadata_empty_run(experiment):
    """Test dsi.retrieve_metadata for an empty run"""
    t_before_run_init = time.perf_counter()

    guid = generate_guid()
    conn = experiment.conn
    dsi = SqliteStorageInterface(guid, conn=conn)

    with pytest.raises(ValueError):
        dsi.retrieve_meta_data()

    dsi.create_run()

    md = dsi.retrieve_meta_data()

    assert md is not None
    assert isinstance(md, MetaData)
    assert None is md.run_completed
    assert RunDescriber(InterDependencies()) == md.run_description
    assert md.run_started > t_before_run_init
    assert None is md.snapshot
    assert {} == md.tags
    assert 1 == md.tier
    assert 'dataset' == md.name
    assert experiment.name == md.exp_name
    assert experiment.sample_name == md.sample_name


def test_retrieve_metadata_various_runs_with_various_metadatas():
    pytest.xfail('not implemented yet')


def test_store_meta_data__run_completed(experiment):
    guid = generate_guid()
    conn = experiment.conn
    dsi = SqliteStorageInterface(guid, conn=conn)
    dsi.create_run()

    control_conn = connect(experiment.path_to_db)

    # assert initial state

    check = "SELECT completed_timestamp,is_completed FROM runs WHERE run_id = ?"
    cursor = control_conn.execute(check, (dsi.run_id,))
    row = cursor.fetchall()[0]
    assert 0 == row['is_completed']
    assert None is row['completed_timestamp']

    # store metadata

    some_time = time.time()
    dsi.store_meta_data(run_completed=some_time)

    # assert metadata was successfully stored

    cursor = control_conn.execute(check, (dsi.run_id,))
    row = cursor.fetchall()[0]
    assert 1 == row['is_completed']
    assert np.allclose(some_time, row['completed_timestamp'])
