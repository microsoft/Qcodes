import re

import pytest

from qcodes.dataset.data_set import DataSet
from qcodes.dataset.sqlite_base import connect
from qcodes.dataset.sqlite_storage_interface import SqliteStorageInterface
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment, dataset)


def test_init_no_guid():
    match_str = re.escape("__init__() missing 1 required"
                          " positional argument: 'guid'")
    with pytest.raises(TypeError, match=match_str):
        SqliteStorageInterface()


def test_init_new_run(experiment):
    pytest.fail('not implemented')

    guid = 'asd'

    conn = connect(':memory:')

    dsi = SqliteStorageInterface(guid, conn=conn)

    assert guid == dsi.guid

    # assert new run created


def test_init_existing_run(experiment):
    pytest.fail('not implemented')

    # create a run in a temp db and get its guid, blah blah
    ds = DataSet()

    guid = ds.guid
    conn = ds.conn

    dsi = SqliteStorageInterface(guid, conn=conn)

    assert guid == dsi.guid

    # assert existing run loaded


def test_retrieve_metadata():
    pytest.fail('not implemented')
