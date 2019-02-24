import os
from contextlib import contextmanager
from copy import deepcopy
import logging
import tempfile
import json

import pytest

import qcodes as qc
from qcodes import new_experiment, new_data_set, ParamSpec
from qcodes.dataset.descriptions import RunDescriber
from qcodes.dataset.dependencies import InterDependencies
from qcodes.dataset.database import (initialise_database,
                                     initialise_or_create_database_at)
# pylint: disable=unused-import
from qcodes.tests.common import error_caused_by
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment,
                                                      temporarily_copied_DB)
from qcodes.dataset.sqlite_base import (connect,
                                        one,
                                        update_GUIDs,
                                        get_db_version_and_newest_available_version,
                                        get_user_version,
                                        set_user_version,
                                        atomic_transaction,
                                        perform_db_upgrade_0_to_1,
                                        perform_db_upgrade_1_to_2,
                                        perform_db_upgrade_2_to_3,
                                        perform_db_upgrade_3_to_4,
                                        _latest_available_version)

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


LATEST_VERSION = _latest_available_version()
VERSIONS = tuple(range(LATEST_VERSION + 1))
LATEST_VERSION_ARG = -1


@pytest.mark.parametrize('ver', VERSIONS + (LATEST_VERSION_ARG,))
def test_connect_upgrades_user_version(ver):
    expected_version = ver if ver != LATEST_VERSION_ARG else LATEST_VERSION
    conn = connect(':memory:', version=ver)
    assert expected_version == get_user_version(conn)


@pytest.mark.parametrize('version', VERSIONS + (LATEST_VERSION_ARG,))
def test_tables_exist(empty_temp_db, version):
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

    dbname_old = os.path.join(v0fixpath, 'empty.db')

    if not os.path.exists(dbname_old):
        pytest.skip("No db-file fixtures found. You can generate test db-files"
                    " using the scripts in the "
                    "https://github.com/QCoDeS/qcodes_generate_test_db/ repo")

    with temporarily_copied_DB(dbname_old, debug=False, version=0) as conn:

        assert get_user_version(conn) == 0

        guid_table_query = "SELECT guid FROM runs"

        with pytest.raises(RuntimeError) as excinfo:
            atomic_transaction(conn, guid_table_query)

        assert error_caused_by(excinfo, 'no such column: guid')

        perform_db_upgrade_0_to_1(conn)
        assert get_user_version(conn) == 1

        c = atomic_transaction(conn, guid_table_query)
        assert len(c.fetchall()) == 0


def test_perform_actual_upgrade_1_to_2():

    v1fixpath = os.path.join(fixturepath, 'db_files', 'version1')

    dbname_old = os.path.join(v1fixpath, 'empty.db')

    if not os.path.exists(dbname_old):
        pytest.skip("No db-file fixtures found. You can generate test db-files"
                    " using the scripts in the legacy_DB_generation folder")

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


def test_perform_actual_upgrade_2_to_3_empty():

    v2fixpath = os.path.join(fixturepath, 'db_files', 'version2')

    dbname_old = os.path.join(v2fixpath, 'empty.db')

    if not os.path.exists(dbname_old):
        pytest.skip("No db-file fixtures found. You can generate test db-files"
                    " using the scripts in the "
                    "https://github.com/QCoDeS/qcodes_generate_test_db/ repo")

    with temporarily_copied_DB(dbname_old, debug=False, version=2) as conn:

        assert get_user_version(conn) == 2

        desc_query = 'SELECT run_description FROM runs'

        with pytest.raises(RuntimeError) as excinfo:
            atomic_transaction(conn, desc_query)

        assert error_caused_by(excinfo, 'no such column: run_description')

        perform_db_upgrade_2_to_3(conn)

        assert get_user_version(conn) == 3

        c = atomic_transaction(conn, desc_query)
        assert len(c.fetchall()) == 0


def test_perform_actual_upgrade_2_to_3_empty_runs():

    v2fixpath = os.path.join(fixturepath, 'db_files', 'version2')

    dbname_old = os.path.join(v2fixpath, 'empty_runs.db')

    if not os.path.exists(dbname_old):
        pytest.skip("No db-file fixtures found. You can generate test db-files"
                    " using the scripts in the "
                    "https://github.com/QCoDeS/qcodes_generate_test_db/ repo")

    with temporarily_copied_DB(dbname_old, debug=False, version=2) as conn:

        perform_db_upgrade_2_to_3(conn)


def test_perform_actual_upgrade_2_to_3_some_runs():

    v2fixpath = os.path.join(fixturepath, 'db_files', 'version2')

    dbname_old = os.path.join(v2fixpath, 'some_runs.db')

    if not os.path.exists(dbname_old):
        pytest.skip("No db-file fixtures found. You can generate test db-files"
                    " using the scripts in the"
                    "https://github.com/QCoDeS/qcodes_generate_test_db/ repo")

    with temporarily_copied_DB(dbname_old, debug=False, version=2) as conn:

        assert get_user_version(conn) == 2

        perform_db_upgrade_2_to_3(conn)

        desc_query = 'SELECT run_description FROM runs'

        c = atomic_transaction(conn, desc_query)
        assert len(c.fetchall()) == 10

        # retrieve the json string and recreate the object

        sql = f"""
              SELECT run_description
              FROM runs
              WHERE run_id == 1
              """
        c = atomic_transaction(conn, sql)
        json_str = one(c, 'run_description')

        desc = RunDescriber.from_json(json_str)
        idp = desc.interdeps
        assert isinstance(idp, InterDependencies)

        # here we verify that the dependencies encoded in
        # tests/dataset/legacy_DB_generation/generate_version_2.py
        # are recovered

        p0 = [p for p in idp.paramspecs if p.name == 'p0'][0]
        assert p0.depends_on == ''
        assert p0.depends_on_ == []
        assert p0.inferred_from == ''
        assert p0.inferred_from_ == []
        assert p0.label == "Parameter 0"
        assert p0.unit == "unit 0"

        p1 = [p for p in idp.paramspecs if p.name == 'p1'][0]
        assert p1.depends_on == ''
        assert p1.depends_on_ == []
        assert p1.inferred_from == ''
        assert p1.inferred_from_ == []
        assert p1.label == "Parameter 1"
        assert p1.unit == "unit 1"

        p2 = [p for p in idp.paramspecs if p.name == 'p2'][0]
        assert p2.depends_on == ''
        assert p2.depends_on_ == []
        assert p2.inferred_from == 'p0'
        assert p2.inferred_from_ == ['p0']
        assert p2.label == "Parameter 2"
        assert p2.unit == "unit 2"

        p3 = [p for p in idp.paramspecs if p.name == 'p3'][0]
        assert p3.depends_on == ''
        assert p3.depends_on_ == []
        assert p3.inferred_from == 'p1, p0'
        assert p3.inferred_from_ == ['p1', 'p0']
        assert p3.label == "Parameter 3"
        assert p3.unit == "unit 3"

        p4 = [p for p in idp.paramspecs if p.name == 'p4'][0]
        assert p4.depends_on == 'p2, p3'
        assert p4.depends_on_ == ['p2', 'p3']
        assert p4.inferred_from == ''
        assert p4.inferred_from_ == []
        assert p4.label == "Parameter 4"
        assert p4.unit == "unit 4"

        p5 = [p for p in idp.paramspecs if p.name == 'p5'][0]
        assert p5.depends_on == ''
        assert p5.depends_on_ == []
        assert p5.inferred_from == 'p0'
        assert p5.inferred_from_ == ['p0']
        assert p5.label == "Parameter 5"
        assert p5.unit == "unit 5"


def test_perform_upgrade_v2_v3_to_v4_fixes():
    """
    Test that a db that was upgraded from v2 to v3 with a buggy
    version will be corrected when upgraded to v4.
    """

    v3fixpath = os.path.join(fixturepath, 'db_files', 'version3')

    dbname_old = os.path.join(v3fixpath, 'some_runs_upgraded_2.db')

    if not os.path.exists(dbname_old):
        pytest.skip("No db-file fixtures found. You can generate test db-files"
                    " using the scripts in the"
                    " https://github.com/QCoDeS/qcodes_generate_test_db/ repo")

    with temporarily_copied_DB(dbname_old, debug=False, version=3) as conn:

        assert get_user_version(conn) == 3

        sql = f"""
              SELECT run_description
              FROM runs
              WHERE run_id == 1
              """
        c = atomic_transaction(conn, sql)
        json_str = one(c, 'run_description')

        desc = RunDescriber.from_json(json_str)
        idp = desc.interdeps
        assert isinstance(idp, InterDependencies)

        p0 = [p for p in idp.paramspecs if p.name == 'p0'][0]
        assert p0.depends_on == ''
        assert p0.depends_on_ == []
        assert p0.inferred_from == ''
        assert p0.inferred_from_ == []
        assert p0.label == "Parameter 0"
        assert p0.unit == "unit 0"

        p1 = [p for p in idp.paramspecs if p.name == 'p1'][0]
        assert p1.depends_on == ''
        assert p1.depends_on_ == []
        assert p1.inferred_from == ''
        assert p1.inferred_from_ == []
        assert p1.label == "Parameter 1"
        assert p1.unit == "unit 1"

        p2 = [p for p in idp.paramspecs if p.name == 'p2'][0]
        assert p2.depends_on == ''
        assert p2.depends_on_ == []
        # the 2 lines below are wrong due to the incorrect upgrade from
        # db version 2 to 3
        assert p2.inferred_from == 'p, 0'
        assert p2.inferred_from_ == ['p', '0']
        assert p2.label == "Parameter 2"
        assert p2.unit == "unit 2"

        p3 = [p for p in idp.paramspecs if p.name == 'p3'][0]
        assert p3.depends_on == ''
        assert p3.depends_on_ == []
        # the 2 lines below are wrong due to the incorrect upgrade from
        # db version 2 to 3
        assert p3.inferred_from == 'p, 1, ,,  , p, 0'
        assert p3.inferred_from_ == ['p', '1', ',', ' ', 'p', '0']
        assert p3.label == "Parameter 3"
        assert p3.unit == "unit 3"

        p4 = [p for p in idp.paramspecs if p.name == 'p4'][0]
        assert p4.depends_on == 'p2, p3'
        assert p4.depends_on_ == ['p2', 'p3']
        assert p4.inferred_from == ''
        assert p4.inferred_from_ == []
        assert p4.label == "Parameter 4"
        assert p4.unit == "unit 4"

        p5 = [p for p in idp.paramspecs if p.name == 'p5'][0]
        assert p5.depends_on == ''
        assert p5.depends_on_ == []
        # the 2 lines below are wrong due to the incorrect upgrade from
        # db version 2 to 3. Here the interdep is missing
        assert p5.inferred_from == ''
        assert p5.inferred_from_ == []
        assert p5.label == "Parameter 5"
        assert p5.unit == "unit 5"

        perform_db_upgrade_3_to_4(conn)

        c = atomic_transaction(conn, sql)
        json_str = one(c, 'run_description')

        desc = RunDescriber.from_json(json_str)
        idp = desc.interdeps
        assert isinstance(idp, InterDependencies)

        p0 = [p for p in idp.paramspecs if p.name == 'p0'][0]
        assert p0.depends_on == ''
        assert p0.depends_on_ == []
        assert p0.inferred_from == ''
        assert p0.inferred_from_ == []
        assert p0.label == "Parameter 0"
        assert p0.unit == "unit 0"

        p1 = [p for p in idp.paramspecs if p.name == 'p1'][0]
        assert p1.depends_on == ''
        assert p1.depends_on_ == []
        assert p1.inferred_from == ''
        assert p1.inferred_from_ == []
        assert p1.label == "Parameter 1"
        assert p1.unit == "unit 1"

        p2 = [p for p in idp.paramspecs if p.name == 'p2'][0]
        assert p2.depends_on == ''
        assert p2.depends_on_ == []
        assert p2.inferred_from == 'p0'
        assert p2.inferred_from_ == ['p0']
        assert p2.label == "Parameter 2"
        assert p2.unit == "unit 2"

        p3 = [p for p in idp.paramspecs if p.name == 'p3'][0]
        assert p3.depends_on == ''
        assert p3.depends_on_ == []
        assert p3.inferred_from == 'p1, p0'
        assert p3.inferred_from_ == ['p1', 'p0']
        assert p3.label == "Parameter 3"
        assert p3.unit == "unit 3"

        p4 = [p for p in idp.paramspecs if p.name == 'p4'][0]
        assert p4.depends_on == 'p2, p3'
        assert p4.depends_on_ == ['p2', 'p3']
        assert p4.inferred_from == ''
        assert p4.inferred_from_ == []
        assert p4.label == "Parameter 4"
        assert p4.unit == "unit 4"

        p5 = [p for p in idp.paramspecs if p.name == 'p5'][0]
        assert p5.depends_on == ''
        assert p5.depends_on_ == []
        assert p5.inferred_from == 'p0'
        assert p5.inferred_from_ == ['p0']
        assert p5.label == "Parameter 5"
        assert p5.unit == "unit 5"


def test_perform_upgrade_v3_to_v4():
    """
    Test that a db upgrade from v2 to v4 works correctly.
    """

    v3fixpath = os.path.join(fixturepath, 'db_files', 'version3')

    dbname_old = os.path.join(v3fixpath, 'some_runs_upgraded_2.db')

    if not os.path.exists(dbname_old):
        pytest.skip("No db-file fixtures found. You can generate test db-files"
                    " using the scripts in the "
                    "https://github.com/QCoDeS/qcodes_generate_test_db/ repo")

    with temporarily_copied_DB(dbname_old, debug=False, version=3) as conn:

        assert get_user_version(conn) == 3

        sql = f"""
              SELECT run_description
              FROM runs
              WHERE run_id == 1
              """

        perform_db_upgrade_3_to_4(conn)

        c = atomic_transaction(conn, sql)
        json_str = one(c, 'run_description')

        desc = RunDescriber.from_json(json_str)
        idp = desc.interdeps
        assert isinstance(idp, InterDependencies)

        p0 = [p for p in idp.paramspecs if p.name == 'p0'][0]
        assert p0.depends_on == ''
        assert p0.depends_on_ == []
        assert p0.inferred_from == ''
        assert p0.inferred_from_ == []
        assert p0.label == "Parameter 0"
        assert p0.unit == "unit 0"

        p1 = [p for p in idp.paramspecs if p.name == 'p1'][0]
        assert p1.depends_on == ''
        assert p1.depends_on_ == []
        assert p1.inferred_from == ''
        assert p1.inferred_from_ == []
        assert p1.label == "Parameter 1"
        assert p1.unit == "unit 1"

        p2 = [p for p in idp.paramspecs if p.name == 'p2'][0]
        assert p2.depends_on == ''
        assert p2.depends_on_ == []
        assert p2.inferred_from == 'p0'
        assert p2.inferred_from_ == ['p0']
        assert p2.label == "Parameter 2"
        assert p2.unit == "unit 2"

        p3 = [p for p in idp.paramspecs if p.name == 'p3'][0]
        assert p3.depends_on == ''
        assert p3.depends_on_ == []
        assert p3.inferred_from == 'p1, p0'
        assert p3.inferred_from_ == ['p1', 'p0']
        assert p3.label == "Parameter 3"
        assert p3.unit == "unit 3"

        p4 = [p for p in idp.paramspecs if p.name == 'p4'][0]
        assert p4.depends_on == 'p2, p3'
        assert p4.depends_on_ == ['p2', 'p3']
        assert p4.inferred_from == ''
        assert p4.inferred_from_ == []
        assert p4.label == "Parameter 4"
        assert p4.unit == "unit 4"

        p5 = [p for p in idp.paramspecs if p.name == 'p5'][0]
        assert p5.depends_on == ''
        assert p5.depends_on_ == []
        assert p5.inferred_from == 'p0'
        assert p5.inferred_from_ == ['p0']
        assert p5.label == "Parameter 5"
        assert p5.unit == "unit 5"


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


@pytest.mark.usefixtures("empty_temp_db")
def test_cannot_connect_to_newer_db():
    conn = connect(qc.config["core"]["db_location"],
                   qc.config["core"]["db_debug"])
    current_version = get_user_version(conn)
    set_user_version(conn, current_version+1)
    conn.close()
    err_msg = f'is version {current_version + 1} but this version of QCoDeS ' \
        f'supports up to version {current_version}'
    with pytest.raises(RuntimeError, match=err_msg):
        conn = connect(qc.config["core"]["db_location"],
                       qc.config["core"]["db_debug"])


def test_latest_available_version():
    assert 4 == _latest_available_version()


@pytest.mark.parametrize('version', VERSIONS)
def test_getting_db_version(version):

    fixpath = os.path.join(fixturepath, 'db_files', f'version{version}')

    dbname = os.path.join(fixpath, 'empty.db')

    if not os.path.exists(dbname):
        pytest.skip("No db-file fixtures found. You can generate test db-files"
                    " using the scripts in the "
                    "https://github.com/QCoDeS/qcodes_generate_test_db/ repo")

    (db_v, new_v) = get_db_version_and_newest_available_version(dbname)

    assert db_v == version
    assert new_v == LATEST_VERSION
