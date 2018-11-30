import re

import pytest

from qcodes.dataset.guids import generate_guid
from qcodes.dataset.sqlite_base import create_run
from qcodes.dataset.sqlite_storage_interface import SqliteStorageInterface
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment, dataset)


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

    pytest.fail('not implemented')

    dsi = SqliteStorageInterface(guid, conn=conn, run_id=None)

    assert guid == dsi.guid

    # assert new run created


def test_init_existing_run(experiment):
    """Test initialising dsi for an existing run"""
    conn = experiment.conn
    guid = generate_guid()
    name = "existing-dataset"
    _, run_id, __ = create_run(conn, experiment.exp_id, name, guid)

    pytest.fail('not implemented')

    dsi = SqliteStorageInterface(guid, conn=conn, run_id=run_id)

    assert guid == dsi.guid

    # assert existing run loaded
    metadata = dsi.retrieve_meta_data()
    assert metadata is not None  # ...etc....


def test_retrieve_metadata():
    """Test dsi.retrieve_metadata for various cases"""
    pytest.fail('not implemented')
