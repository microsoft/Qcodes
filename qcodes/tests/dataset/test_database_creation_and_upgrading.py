import os
from contextlib import contextmanager
from copy import deepcopy
import logging
import tempfile

import pytest

import qcodes as qc
from qcodes import new_experiment, new_data_set, ParamSpec
from qcodes.dataset.database import (initialise_database,
                                     initialise_or_create_database_at)
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment,
                                                      temporarily_copied_DB)
from qcodes.dataset.sqlite_base import (connect,
                                        update_GUIDs,
                                        get_user_version,
                                        atomic_transaction,
                                        perform_db_upgrade_0_to_1,
                                        perform_db_upgrade_1_to_2)

from qcodes.dataset.guids import parse_guid
import qcodes.tests.dataset

fixturepath = os.sep.join(qcodes.tests.dataset.__file__.split(os.sep)[:-1])
fixturepath = os.path.join(fixturepath, 'fixtures')


@contextmanager
def location_and_station_set_to(location: int, work_station: int):
    cfg = qc.Config()
    old_cfg = deepcopy(cfg.current_config)
    cfg['GUID_components']['location'] = location
    cfg['GUID_components']['work_station'] = work_station
    cfg.save_to_home()

    try:
        yield

    finally:
        cfg.current_config = old_cfg
        cfg.save_to_home()


@pytest.mark.usefixtures("empty_temp_db")
def test_tables_exist():
    for version in [-1, 0, 1]:
        conn = connect(qc.config["core"]["db_location"],
                       qc.config["core"]["db_debug"],
                       version=version)
        cursor = conn.execute("select sql from sqlite_master"
                              " where type = 'table'")
        expected_tables = ['experiments', 'runs', 'layouts', 'dependencies']
        rows = [row for row in cursor]
        assert len(rows) == len(expected_tables)
        for row, expected_table in zip(rows, expected_tables):
            assert expected_table in row['sql']
        conn.close()


def test_initialise_database_at_for_nonexisting_db():
    with tempfile.TemporaryDirectory() as tmpdirname:
        db_location = os.path.join(tmpdirname, 'temp.db')
        assert not os.path.exists(db_location)

        initialise_or_create_database_at(db_location)

        assert os.path.exists(db_location)
        assert qc.config["core"]["db_location"] == db_location


def test_initialise_database_at_for_existing_db():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Define DB location
        db_location = os.path.join(tmpdirname, 'temp.db')
        assert not os.path.exists(db_location)

        # Create DB file
        qc.config["core"]["db_location"] = db_location
        initialise_database()

        # Check if it has been created correctly
        assert os.path.exists(db_location)
        assert qc.config["core"]["db_location"] == db_location

        # Call function under test
        initialise_or_create_database_at(db_location)

        # Check if the DB is still correct
        assert os.path.exists(db_location)
        assert qc.config["core"]["db_location"] == db_location


def test_perform_actual_upgrade_0_to_1():
    # we cannot use the empty_temp_db, since that has already called connect
    # and is therefore latest version already

    v0fixpath = os.path.join(fixturepath, 'db_files', 'version0')

    if not os.path.exists(v0fixpath):
        pytest.skip("No db-file fixtures found. You can generate test db-files"
                    " using the scripts in the legacy_DB_generation folder")

    dbname_old = os.path.join(v0fixpath, 'empty.db')

    with temporarily_copied_DB(dbname_old, debug=False, version=0) as conn:

        assert get_user_version(conn) == 0

        guid_table_query = "SELECT guid FROM runs"

        with pytest.raises(RuntimeError):
            atomic_transaction(conn, guid_table_query)

        perform_db_upgrade_0_to_1(conn)
        assert get_user_version(conn) == 1

        c = atomic_transaction(conn, guid_table_query)
        assert len(c.fetchall()) == 0


def test_perform_actual_upgrade_1_to_2():

    v1fixpath = os.path.join(fixturepath, 'db_files', 'version1')

    if not os.path.exists(v1fixpath):
        pytest.skip("No db-file fixtures found. You can generate test db-files"
                    " using the scripts in the legacy_DB_generation folder")

    dbname_old = os.path.join(v1fixpath, 'empty.db')

    with temporarily_copied_DB(dbname_old, debug=False, version=1) as conn:

        assert get_user_version(conn) == 1

        guid_table_query = "SELECT guid FROM runs"

        c = atomic_transaction(conn, guid_table_query)
        assert len(c.fetchall()) == 0

        index_query = "PRAGMA index_list(runs)"

        c = atomic_transaction(conn, index_query)
        assert len(c.fetchall()) == 0

        perform_db_upgrade_1_to_2(conn)

        c = atomic_transaction(conn, index_query)
        assert len(c.fetchall()) == 2


@pytest.mark.usefixtures("empty_temp_db")
def test_update_existing_guids(caplog):

    old_loc = 101
    old_ws = 1200

    new_loc = 232
    new_ws = 52123

    # prepare five runs with different location and work station codes

    with location_and_station_set_to(0, 0):
        new_experiment('test', sample_name='test_sample')

        ds1 = new_data_set('ds_one')
        xparam = ParamSpec('x', 'numeric')
        ds1.add_parameter(xparam)
        ds1.add_result({'x': 1})

        ds2 = new_data_set('ds_two')
        ds2.add_parameter(xparam)
        ds2.add_result({'x': 2})

        guid_comps_1 = parse_guid(ds1.guid)
        assert guid_comps_1['location'] == 0
        assert guid_comps_1['work_station'] == 0

        guid_comps_2 = parse_guid(ds2.guid)
        assert guid_comps_2['location'] == 0
        assert guid_comps_2['work_station'] == 0

    with location_and_station_set_to(0, old_ws):
        ds3 = new_data_set('ds_three')
        xparam = ParamSpec('x', 'numeric')
        ds3.add_parameter(xparam)
        ds3.add_result({'x': 3})

    with location_and_station_set_to(old_loc, 0):
        ds4 = new_data_set('ds_four')
        xparam = ParamSpec('x', 'numeric')
        ds4.add_parameter(xparam)
        ds4.add_result({'x': 4})

    with location_and_station_set_to(old_loc, old_ws):
        ds5 = new_data_set('ds_five')
        xparam = ParamSpec('x', 'numeric')
        ds5.add_parameter(xparam)
        ds5.add_result({'x': 5})

    with location_and_station_set_to(new_loc, new_ws):

        caplog.clear()
        expected_levels = ['INFO',
                           'INFO', 'INFO',
                           'INFO', 'INFO',
                           'INFO', 'WARNING',
                           'INFO', 'WARNING',
                           'INFO', 'INFO']

        with caplog.at_level(logging.INFO):
            update_GUIDs(ds1.conn)

            for record, lvl in zip(caplog.records, expected_levels):
                assert record.levelname == lvl

        guid_comps_1 = parse_guid(ds1.guid)
        assert guid_comps_1['location'] == new_loc
        assert guid_comps_1['work_station'] == new_ws

        guid_comps_2 = parse_guid(ds2.guid)
        assert guid_comps_2['location'] == new_loc
        assert guid_comps_2['work_station'] == new_ws

        guid_comps_3 = parse_guid(ds3.guid)
        assert guid_comps_3['location'] == 0
        assert guid_comps_3['work_station'] == old_ws

        guid_comps_4 = parse_guid(ds4.guid)
        assert guid_comps_4['location'] == old_loc
        assert guid_comps_4['work_station'] == 0

        guid_comps_5 = parse_guid(ds5.guid)
        assert guid_comps_5['location'] == old_loc
        assert guid_comps_5['work_station'] == old_ws
