import re
import time

import pytest

from qcodes.dataset.data_storage_interface import MetaData
from qcodes.dataset.dependencies import InterDependencies
from qcodes.dataset.descriptions import RunDescriber
from qcodes.dataset.guids import generate_guid
from qcodes.dataset.sqlite_base import create_run, get_runs
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


def test_init_new_run(experiment):
    """Test initialising dsi for a new run"""
    conn = experiment.conn
    guid = generate_guid()

    # ensure there are no runs in the database
    assert [] == get_runs(conn)

    dsi = SqliteStorageInterface(guid, conn=conn, run_id=None)

    assert experiment.conn is dsi.conn
    assert guid == dsi.guid
    assert 1 == dsi.run_id
    assert experiment.path_to_db == dsi._path_to_db

    # assert new run created
    runs_rows = get_runs(conn)
    assert 1 == len(runs_rows)
    assert 1 == runs_rows[0]['run_id']

    md = dsi.retrieve_meta_data()

    assert None is md.run_completed
    assert RunDescriber(InterDependencies()) == md.run_description
    assert md.run_started > 0
    assert None is md.snapshot
    assert {} == md.tags
    assert 1 == md.tier

    pytest.xfail('more assertions on the fact that run is created are needed')


def test_init_existing_run(experiment):
    """Test initialising dsi for an existing run"""
    conn = experiment.conn
    guid = generate_guid()
    name = "existing-dataset"
    _, run_id, __ = create_run(conn, experiment.exp_id, name, guid)

    dsi = SqliteStorageInterface(guid, conn=conn, run_id=run_id)

    assert experiment.conn is dsi.conn
    assert guid == dsi.guid
    assert run_id == dsi.run_id
    assert experiment.path_to_db == dsi._path_to_db

    # assert existing run loaded

    md = dsi.retrieve_meta_data()

    assert None is md.run_completed
    assert RunDescriber(InterDependencies()) == md.run_description
    assert md.run_started > 0
    assert None is md.snapshot
    assert {} == md.tags
    assert 1 == md.tier

    pytest.xfail(
        'more assertions needed for the fact that we loaded existing runs')


def test_retrieve_metadata_empty_run(experiment):
    """Test dsi.retrieve_metadata for an empty run"""
    t_before_run_init = time.perf_counter()

    guid = generate_guid()
    conn = experiment.conn
    dsi = SqliteStorageInterface(guid, conn=conn, run_id=None)

    md = dsi.retrieve_meta_data()

    assert md is not None
    assert isinstance(md, MetaData)
    assert None is md.run_completed
    assert RunDescriber(InterDependencies()) == md.run_description
    assert md.run_started > t_before_run_init
    assert None is md.snapshot
    assert {} == md.tags
    assert 1 == md.tier


def test_retrieve_metadata_various_runs_with_various_metadatas():
    pytest.xfail('not implemented yet')
