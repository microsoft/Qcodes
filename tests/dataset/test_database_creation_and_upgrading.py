import json
import logging
import os
from contextlib import contextmanager
from copy import deepcopy

import pytest
from pytest import LogCaptureFixture

import qcodes as qc
import qcodes.dataset.descriptions.versioning.serialization as serial
import tests.dataset
from qcodes.dataset import (
    ConnectionPlus,
    connect,
    initialise_database,
    initialise_or_create_database_at,
    load_by_counter,
    load_by_id,
    load_by_run_spec,
    new_data_set,
    new_experiment,
)
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.descriptions.versioning.v0 import InterDependencies
from qcodes.dataset.guids import parse_guid
from qcodes.dataset.sqlite.connection import atomic_transaction
from qcodes.dataset.sqlite.database import get_db_version_and_newest_available_version
from qcodes.dataset.sqlite.db_upgrades import (
    _latest_available_version,
    perform_db_upgrade,
    perform_db_upgrade_0_to_1,
    perform_db_upgrade_1_to_2,
    perform_db_upgrade_2_to_3,
    perform_db_upgrade_3_to_4,
    perform_db_upgrade_4_to_5,
    perform_db_upgrade_5_to_6,
    perform_db_upgrade_6_to_7,
    perform_db_upgrade_7_to_8,
    perform_db_upgrade_8_to_9,
)
from qcodes.dataset.sqlite.db_upgrades.version import get_user_version, set_user_version
from qcodes.dataset.sqlite.queries import get_run_description, update_GUIDs
from qcodes.dataset.sqlite.query_helpers import (
    get_description_map,
    is_column_in_table,
    one,
)
from tests.common import error_caused_by, skip_if_no_fixtures
from tests.dataset.conftest import temporarily_copied_DB

fixturepath = os.sep.join(tests.dataset.__file__.split(os.sep)[:-1])
fixturepath = os.path.join(fixturepath, "fixtures")


@contextmanager
def location_and_station_set_to(location: int, work_station: int):
    cfg = qc.config.current_config
    if cfg is None:
        raise RuntimeError("Expected config to be not None.")
    old_cfg = deepcopy(cfg)
    cfg["GUID_components"]["location"] = location
    cfg["GUID_components"]["work_station"] = work_station

    try:
        yield

    finally:
        qc.config.current_config = old_cfg


LATEST_VERSION = _latest_available_version()
VERSIONS = tuple(range(LATEST_VERSION + 1))
LATEST_VERSION_ARG = -1


@pytest.mark.parametrize("ver", VERSIONS + (LATEST_VERSION_ARG,))
def test_connect_upgrades_user_version(ver) -> None:
    expected_version = ver if ver != LATEST_VERSION_ARG else LATEST_VERSION
    conn = connect(":memory:", version=ver)
    assert expected_version == get_user_version(conn)


@pytest.mark.parametrize("version", VERSIONS + (LATEST_VERSION_ARG,))
def test_tables_exist(empty_temp_db, version) -> None:
    conn = connect(
        qc.config["core"]["db_location"], qc.config["core"]["db_debug"], version=version
    )
    query = """
    SELECT sql FROM sqlite_master
    WHERE type = 'table'
    """
    cursor = conn.execute(query)
    expected_tables = ["experiments", "runs", "layouts", "dependencies"]
    rows = [row for row in cursor]
    assert len(rows) == len(expected_tables)
    for (sql,), expected_table in zip(rows, expected_tables):
        assert expected_table in sql
    conn.close()


def test_initialise_database_at_for_nonexisting_db(tmp_path) -> None:
    db_location = str(tmp_path / "temp.db")
    assert not os.path.exists(db_location)

    initialise_or_create_database_at(db_location)

    assert os.path.exists(db_location)
    assert qc.config["core"]["db_location"] == db_location


def test_initialise_database_at_for_nonexisting_db_pathlib_path(tmp_path) -> None:
    db_location = tmp_path / "temp.db"
    assert not db_location.exists()

    initialise_or_create_database_at(db_location)

    assert db_location.exists()
    assert qc.config["core"]["db_location"] == str(db_location)


def test_initialise_database_at_for_existing_db(tmp_path) -> None:
    # Define DB location
    db_location = str(tmp_path / "temp.db")
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


def test_perform_actual_upgrade_0_to_1() -> None:
    # we cannot use the empty_temp_db, since that has already called connect
    # and is therefore latest version already

    v0fixpath = os.path.join(fixturepath, "db_files", "version0")

    dbname_old = os.path.join(v0fixpath, "empty.db")

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=0) as conn:
        assert get_user_version(conn) == 0

        guid_table_query = "SELECT guid FROM runs"

        with pytest.raises(RuntimeError) as excinfo:
            atomic_transaction(conn, guid_table_query)

        assert error_caused_by(excinfo, "no such column: guid")

        perform_db_upgrade_0_to_1(conn)
        assert get_user_version(conn) == 1

        c = atomic_transaction(conn, guid_table_query)
        assert len(c.fetchall()) == 0


def test_perform_actual_upgrade_1_to_2() -> None:
    v1fixpath = os.path.join(fixturepath, "db_files", "version1")

    dbname_old = os.path.join(v1fixpath, "empty.db")

    skip_if_no_fixtures(dbname_old)

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


def test_perform_actual_upgrade_2_to_3_empty() -> None:
    v2fixpath = os.path.join(fixturepath, "db_files", "version2")

    dbname_old = os.path.join(v2fixpath, "empty.db")

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=2) as conn:
        assert get_user_version(conn) == 2

        desc_query = "SELECT run_description FROM runs"

        with pytest.raises(RuntimeError) as excinfo:
            atomic_transaction(conn, desc_query)

        assert error_caused_by(excinfo, "no such column: run_description")

        perform_db_upgrade_2_to_3(conn)

        assert get_user_version(conn) == 3

        c = atomic_transaction(conn, desc_query)
        assert len(c.fetchall()) == 0


def test_perform_actual_upgrade_2_to_3_empty_runs() -> None:
    v2fixpath = os.path.join(fixturepath, "db_files", "version2")

    dbname_old = os.path.join(v2fixpath, "empty_runs.db")

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=2) as conn:
        perform_db_upgrade_2_to_3(conn)


def test_perform_actual_upgrade_2_to_3_some_runs() -> None:
    v2fixpath = os.path.join(fixturepath, "db_files", "version2")

    dbname_old = os.path.join(v2fixpath, "some_runs.db")

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=2) as conn:
        assert get_user_version(conn) == 2

        perform_db_upgrade_2_to_3(conn)

        desc_query = "SELECT run_description FROM runs"

        c = atomic_transaction(conn, desc_query)
        assert len(c.fetchall()) == 10

        # retrieve the json string and recreate the object

        sql = """
              SELECT run_description
              FROM runs
              WHERE run_id == 1
              """
        c = atomic_transaction(conn, sql)
        json_str = one(c, "run_description")

        unversioned_dict = json.loads(json_str)
        idp = InterDependencies._from_dict(unversioned_dict["interdependencies"])
        assert isinstance(idp, InterDependencies)

        # here we verify that the dependencies encoded in
        # tests/dataset/legacy_DB_generation/generate_version_2.py
        # are recovered

        p0 = [p for p in idp.paramspecs if p.name == "p0"][0]
        assert p0.depends_on == ""
        assert p0.depends_on_ == []
        assert p0.inferred_from == ""
        assert p0.inferred_from_ == []
        assert p0.label == "Parameter 0"
        assert p0.unit == "unit 0"

        p1 = [p for p in idp.paramspecs if p.name == "p1"][0]
        assert p1.depends_on == ""
        assert p1.depends_on_ == []
        assert p1.inferred_from == ""
        assert p1.inferred_from_ == []
        assert p1.label == "Parameter 1"
        assert p1.unit == "unit 1"

        p2 = [p for p in idp.paramspecs if p.name == "p2"][0]
        assert p2.depends_on == ""
        assert p2.depends_on_ == []
        assert p2.inferred_from == "p0"
        assert p2.inferred_from_ == ["p0"]
        assert p2.label == "Parameter 2"
        assert p2.unit == "unit 2"

        p3 = [p for p in idp.paramspecs if p.name == "p3"][0]
        assert p3.depends_on == ""
        assert p3.depends_on_ == []
        assert p3.inferred_from == "p1, p0"
        assert p3.inferred_from_ == ["p1", "p0"]
        assert p3.label == "Parameter 3"
        assert p3.unit == "unit 3"

        p4 = [p for p in idp.paramspecs if p.name == "p4"][0]
        assert p4.depends_on == "p2, p3"
        assert p4.depends_on_ == ["p2", "p3"]
        assert p4.inferred_from == ""
        assert p4.inferred_from_ == []
        assert p4.label == "Parameter 4"
        assert p4.unit == "unit 4"

        p5 = [p for p in idp.paramspecs if p.name == "p5"][0]
        assert p5.depends_on == ""
        assert p5.depends_on_ == []
        assert p5.inferred_from == "p0"
        assert p5.inferred_from_ == ["p0"]
        assert p5.label == "Parameter 5"
        assert p5.unit == "unit 5"


def test_perform_upgrade_v2_v3_to_v4_fixes() -> None:
    """
    Test that a db that was upgraded from v2 to v3 with a buggy
    version will be corrected when upgraded to v4.
    """

    v3fixpath = os.path.join(fixturepath, "db_files", "version3")

    dbname_old = os.path.join(v3fixpath, "some_runs_upgraded_2.db")

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=3) as conn:
        assert get_user_version(conn) == 3

        sql = """
              SELECT run_description
              FROM runs
              WHERE run_id == 1
              """
        c = atomic_transaction(conn, sql)
        json_str = one(c, "run_description")

        unversioned_dict = json.loads(json_str)
        idp = InterDependencies._from_dict(unversioned_dict["interdependencies"])

        assert isinstance(idp, InterDependencies)

        p0 = [p for p in idp.paramspecs if p.name == "p0"][0]
        assert p0.depends_on == ""
        assert p0.depends_on_ == []
        assert p0.inferred_from == ""
        assert p0.inferred_from_ == []
        assert p0.label == "Parameter 0"
        assert p0.unit == "unit 0"

        p1 = [p for p in idp.paramspecs if p.name == "p1"][0]
        assert p1.depends_on == ""
        assert p1.depends_on_ == []
        assert p1.inferred_from == ""
        assert p1.inferred_from_ == []
        assert p1.label == "Parameter 1"
        assert p1.unit == "unit 1"

        p2 = [p for p in idp.paramspecs if p.name == "p2"][0]
        assert p2.depends_on == ""
        assert p2.depends_on_ == []
        # the 2 lines below are wrong due to the incorrect upgrade from
        # db version 2 to 3
        assert p2.inferred_from == "p, 0"
        assert p2.inferred_from_ == ["p", "0"]
        assert p2.label == "Parameter 2"
        assert p2.unit == "unit 2"

        p3 = [p for p in idp.paramspecs if p.name == "p3"][0]
        assert p3.depends_on == ""
        assert p3.depends_on_ == []
        # the 2 lines below are wrong due to the incorrect upgrade from
        # db version 2 to 3
        assert p3.inferred_from == "p, 1, ,,  , p, 0"
        assert p3.inferred_from_ == ["p", "1", ",", " ", "p", "0"]
        assert p3.label == "Parameter 3"
        assert p3.unit == "unit 3"

        p4 = [p for p in idp.paramspecs if p.name == "p4"][0]
        assert p4.depends_on == "p2, p3"
        assert p4.depends_on_ == ["p2", "p3"]
        assert p4.inferred_from == ""
        assert p4.inferred_from_ == []
        assert p4.label == "Parameter 4"
        assert p4.unit == "unit 4"

        p5 = [p for p in idp.paramspecs if p.name == "p5"][0]
        assert p5.depends_on == ""
        assert p5.depends_on_ == []
        # the 2 lines below are wrong due to the incorrect upgrade from
        # db version 2 to 3. Here the interdep is missing
        assert p5.inferred_from == ""
        assert p5.inferred_from_ == []
        assert p5.label == "Parameter 5"
        assert p5.unit == "unit 5"

        perform_db_upgrade_3_to_4(conn)

        c = atomic_transaction(conn, sql)
        json_str = one(c, "run_description")

        unversioned_dict = json.loads(json_str)
        idp = InterDependencies._from_dict(unversioned_dict["interdependencies"])

        assert isinstance(idp, InterDependencies)

        p0 = [p for p in idp.paramspecs if p.name == "p0"][0]
        assert p0.depends_on == ""
        assert p0.depends_on_ == []
        assert p0.inferred_from == ""
        assert p0.inferred_from_ == []
        assert p0.label == "Parameter 0"
        assert p0.unit == "unit 0"

        p1 = [p for p in idp.paramspecs if p.name == "p1"][0]
        assert p1.depends_on == ""
        assert p1.depends_on_ == []
        assert p1.inferred_from == ""
        assert p1.inferred_from_ == []
        assert p1.label == "Parameter 1"
        assert p1.unit == "unit 1"

        p2 = [p for p in idp.paramspecs if p.name == "p2"][0]
        assert p2.depends_on == ""
        assert p2.depends_on_ == []
        assert p2.inferred_from == "p0"
        assert p2.inferred_from_ == ["p0"]
        assert p2.label == "Parameter 2"
        assert p2.unit == "unit 2"

        p3 = [p for p in idp.paramspecs if p.name == "p3"][0]
        assert p3.depends_on == ""
        assert p3.depends_on_ == []
        assert p3.inferred_from == "p1, p0"
        assert p3.inferred_from_ == ["p1", "p0"]
        assert p3.label == "Parameter 3"
        assert p3.unit == "unit 3"

        p4 = [p for p in idp.paramspecs if p.name == "p4"][0]
        assert p4.depends_on == "p2, p3"
        assert p4.depends_on_ == ["p2", "p3"]
        assert p4.inferred_from == ""
        assert p4.inferred_from_ == []
        assert p4.label == "Parameter 4"
        assert p4.unit == "unit 4"

        p5 = [p for p in idp.paramspecs if p.name == "p5"][0]
        assert p5.depends_on == ""
        assert p5.depends_on_ == []
        assert p5.inferred_from == "p0"
        assert p5.inferred_from_ == ["p0"]
        assert p5.label == "Parameter 5"
        assert p5.unit == "unit 5"


def test_perform_upgrade_v3_to_v4() -> None:
    """
    Test that a db upgrade from v2 to v4 works correctly.
    """

    v3fixpath = os.path.join(fixturepath, "db_files", "version3")

    dbname_old = os.path.join(v3fixpath, "some_runs_upgraded_2.db")

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=3) as conn:
        assert get_user_version(conn) == 3

        sql = """
              SELECT run_description
              FROM runs
              WHERE run_id == 1
              """

        perform_db_upgrade_3_to_4(conn)

        c = atomic_transaction(conn, sql)
        json_str = one(c, "run_description")

        unversioned_dict = json.loads(json_str)
        idp = InterDependencies._from_dict(unversioned_dict["interdependencies"])

        assert isinstance(idp, InterDependencies)

        p0 = [p for p in idp.paramspecs if p.name == "p0"][0]
        assert p0.depends_on == ""
        assert p0.depends_on_ == []
        assert p0.inferred_from == ""
        assert p0.inferred_from_ == []
        assert p0.label == "Parameter 0"
        assert p0.unit == "unit 0"

        p1 = [p for p in idp.paramspecs if p.name == "p1"][0]
        assert p1.depends_on == ""
        assert p1.depends_on_ == []
        assert p1.inferred_from == ""
        assert p1.inferred_from_ == []
        assert p1.label == "Parameter 1"
        assert p1.unit == "unit 1"

        p2 = [p for p in idp.paramspecs if p.name == "p2"][0]
        assert p2.depends_on == ""
        assert p2.depends_on_ == []
        assert p2.inferred_from == "p0"
        assert p2.inferred_from_ == ["p0"]
        assert p2.label == "Parameter 2"
        assert p2.unit == "unit 2"

        p3 = [p for p in idp.paramspecs if p.name == "p3"][0]
        assert p3.depends_on == ""
        assert p3.depends_on_ == []
        assert p3.inferred_from == "p1, p0"
        assert p3.inferred_from_ == ["p1", "p0"]
        assert p3.label == "Parameter 3"
        assert p3.unit == "unit 3"

        p4 = [p for p in idp.paramspecs if p.name == "p4"][0]
        assert p4.depends_on == "p2, p3"
        assert p4.depends_on_ == ["p2", "p3"]
        assert p4.inferred_from == ""
        assert p4.inferred_from_ == []
        assert p4.label == "Parameter 4"
        assert p4.unit == "unit 4"

        p5 = [p for p in idp.paramspecs if p.name == "p5"][0]
        assert p5.depends_on == ""
        assert p5.depends_on_ == []
        assert p5.inferred_from == "p0"
        assert p5.inferred_from_ == ["p0"]
        assert p5.label == "Parameter 5"
        assert p5.unit == "unit 5"


@pytest.mark.usefixtures("empty_temp_db")
def test_update_existing_guids(caplog: LogCaptureFixture) -> None:
    old_loc = 101
    old_ws = 1200

    new_loc = 232
    new_ws = 52123

    # prepare five runs with different location and work station codes

    with location_and_station_set_to(0, 0):
        new_experiment("test", sample_name="test_sample")

        ds1 = new_data_set("ds_one")
        xparam = ParamSpecBase("x", "numeric")
        idps = InterDependencies_(standalones=(xparam,))
        ds1.set_interdependencies(idps)
        ds1.mark_started()
        ds1.add_results([{"x": 1}])

        ds2 = new_data_set("ds_two")
        ds2.set_interdependencies(idps)
        ds2.mark_started()
        ds2.add_results([{"x": 2}])

        _assert_loc_station(ds1, 0, 0)
        _assert_loc_station(ds2, 0, 0)

    with location_and_station_set_to(0, old_ws):
        ds3 = new_data_set("ds_three")
        ds3.set_interdependencies(idps)
        ds3.mark_started()
        ds3.add_results([{"x": 3}])

        _assert_loc_station(ds3, 0, old_ws)

    with location_and_station_set_to(old_loc, 0):
        ds4 = new_data_set("ds_four")
        ds4.set_interdependencies(idps)
        ds4.mark_started()
        ds4.add_results([{"x": 4}])

        _assert_loc_station(ds4, old_loc, 0)

    with location_and_station_set_to(old_loc, old_ws):
        ds5 = new_data_set("ds_five")
        ds5.set_interdependencies(idps)
        ds5.mark_started()
        ds5.add_results([{"x": 5}])

        _assert_loc_station(ds5, old_loc, old_ws)

    with location_and_station_set_to(new_loc, new_ws):
        caplog.clear()
        expected_levels = [
            "INFO",
            "INFO",
            "INFO",
            "INFO",
            "INFO",
            "INFO",
            "WARNING",
            "INFO",
            "WARNING",
            "INFO",
            "INFO",
        ]

        with caplog.at_level(logging.INFO):
            update_GUIDs(ds1.conn)

            for record, lvl in zip(caplog.records, expected_levels):
                print(record)
                assert record.levelname == lvl

        _assert_loc_station(ds1, new_loc, new_ws)
        _assert_loc_station(ds2, new_loc, new_ws)
        _assert_loc_station(ds3, 0, old_ws)
        _assert_loc_station(ds4, old_loc, 0)
        _assert_loc_station(ds5, old_loc, old_ws)


def _assert_loc_station(ds, expected_loc, expected_station):
    guid_dict = parse_guid(ds.guid)
    assert guid_dict["location"] == expected_loc
    assert guid_dict["work_station"] == expected_station


@pytest.mark.parametrize(
    "db_file", ["empty", "with_runs_but_no_snapshots", "with_runs_and_snapshots"]
)
def test_perform_actual_upgrade_4_to_5(db_file) -> None:
    v4fixpath = os.path.join(fixturepath, "db_files", "version4")

    db_file += ".db"
    dbname_old = os.path.join(v4fixpath, db_file)

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=4) as conn:
        # firstly, assert the situation with 'snapshot' column of 'runs' table
        if "with_runs_and_snapshots" in db_file:
            assert is_column_in_table(conn, "runs", "snapshot")
        else:
            assert not is_column_in_table(conn, "runs", "snapshot")

        # secondly, perform the upgrade
        perform_db_upgrade_4_to_5(conn)

        # finally, assert the 'snapshot' column exists in 'runs' table
        assert is_column_in_table(conn, "runs", "snapshot")


def test_perform_actual_upgrade_5_to_6() -> None:
    fixpath = os.path.join(fixturepath, "db_files", "version5")

    db_file = "empty.db"
    dbname_old = os.path.join(fixpath, db_file)

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=5) as conn:
        perform_db_upgrade_5_to_6(conn)
        assert get_user_version(conn) == 6

    db_file = "some_runs.db"
    dbname_old = os.path.join(fixpath, db_file)

    with temporarily_copied_DB(dbname_old, debug=False, version=5) as conn:
        perform_db_upgrade_5_to_6(conn)
        assert get_user_version(conn) == 6

        no_of_runs_query = "SELECT max(run_id) FROM runs"
        no_of_runs = one(atomic_transaction(conn, no_of_runs_query), "max(run_id)")
        assert no_of_runs == 10

        for run_id in range(1, no_of_runs + 1):
            json_str = get_run_description(conn, run_id)

            deser = json.loads(json_str)
            assert deser["version"] == 0

            desc = serial.from_json_to_current(json_str)
            assert desc._version == 3


def test_perform_upgrade_6_7() -> None:
    fixpath = os.path.join(fixturepath, "db_files", "version6")

    db_file = "empty.db"
    dbname_old = os.path.join(fixpath, db_file)

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=6) as conn:
        perform_db_upgrade_6_to_7(conn)
        assert get_user_version(conn) == 7


def test_perform_actual_upgrade_6_to_7() -> None:
    fixpath = os.path.join(fixturepath, "db_files", "version6")

    db_file = "some_runs.db"
    dbname_old = os.path.join(fixpath, db_file)

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=6) as conn:
        assert isinstance(conn, ConnectionPlus)
        perform_db_upgrade_6_to_7(conn)
        assert get_user_version(conn) == 7

        no_of_runs_query = "SELECT max(run_id) FROM runs"
        no_of_runs = one(atomic_transaction(conn, no_of_runs_query), "max(run_id)")
        assert no_of_runs == 10

        c = atomic_transaction(conn, "PRAGMA table_info(runs)")
        description = get_description_map(c)
        columns = c.fetchall()
        col_names = [col[description["name"]] for col in columns]

        assert "captured_run_id" in col_names
        assert "captured_counter" in col_names

        for run_id in range(1, no_of_runs + 1):
            ds1 = load_by_id(run_id, conn)
            ds2 = load_by_run_spec(captured_run_id=run_id, conn=conn)

            assert isinstance(ds1, DataSet)
            assert ds1.the_same_dataset_as(ds2)

            assert ds1.run_id == run_id
            assert ds1.run_id == ds1.captured_run_id
            assert ds2.run_id == run_id
            assert ds2.run_id == ds2.captured_run_id

        exp_id = 1
        for counter in range(1, no_of_runs + 1):
            ds1 = load_by_counter(counter, exp_id, conn)
            ds2 = load_by_run_spec(captured_counter=counter, conn=conn)

            assert isinstance(ds1, DataSet)
            assert ds1.the_same_dataset_as(ds2)
            assert ds1.counter == counter
            assert ds1.counter == ds1.captured_counter
            assert ds2.counter == counter
            assert ds2.counter == ds2.captured_counter


def test_perform_actual_upgrade_6_to_newest_add_new_data() -> None:
    """
    Insert new runs on top of existing runs upgraded and verify that they
    get the correct captured_run_id and captured_counter
    """
    import numpy as np

    from qcodes.dataset.measurements import Measurement
    from qcodes.parameters import Parameter

    fixpath = os.path.join(fixturepath, "db_files", "version6")

    db_file = "some_runs.db"
    dbname_old = os.path.join(fixpath, db_file)

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=6) as conn:
        assert isinstance(conn, ConnectionPlus)
        perform_db_upgrade(conn)
        assert get_user_version(conn) >= 7
        no_of_runs_query = "SELECT max(run_id) FROM runs"
        no_of_runs = one(atomic_transaction(conn, no_of_runs_query), "max(run_id)")

        # Now let's insert new runs and ensure that they also get
        # captured_run_id assigned.
        params = []
        for n in range(5):
            params.append(
                Parameter(
                    f"p{n}",
                    label=f"Parameter {n}",
                    unit=f"unit {n}",
                    set_cmd=None,
                    get_cmd=None,
                )
            )

        # Set up an experiment
        exp = new_experiment("some-exp", "some-sample", conn=conn)
        meas = Measurement(exp=exp)
        meas.register_parameter(params[0])
        meas.register_parameter(params[1])
        meas.register_parameter(params[2], basis=(params[0],))
        meas.register_parameter(params[3], basis=(params[1],))
        meas.register_parameter(params[4], setpoints=(params[2], params[3]))

        # Make a number of identical runs
        for _ in range(10):
            with meas.run() as datasaver:
                for x in np.random.rand(10):
                    for y in np.random.rand(10):
                        z = np.random.rand()
                        datasaver.add_result(
                            (params[0], 0),
                            (params[1], 1),
                            (params[2], x),
                            (params[3], y),
                            (params[4], z),
                        )

        no_of_runs_new = one(atomic_transaction(conn, no_of_runs_query), "max(run_id)")
        assert no_of_runs_new == 20

        # check that run_id is equivalent to captured_run_id for new
        # runs
        for run_id in range(no_of_runs, no_of_runs_new + 1):
            ds1 = load_by_id(run_id, conn)
            ds2 = load_by_run_spec(captured_run_id=run_id, conn=conn)

            assert isinstance(ds1, DataSet)
            assert ds1.the_same_dataset_as(ds2)

            assert ds1.run_id == run_id
            assert ds1.run_id == ds1.captured_run_id
            assert ds2.run_id == run_id
            assert ds2.run_id == ds2.captured_run_id

        # we are creating a new experiment into a db with one exp so:
        exp_id = 2

        # check that counter is equivalent to captured_counter for new
        # runs
        for counter in range(1, no_of_runs_new - no_of_runs + 1):
            ds1 = load_by_counter(counter, exp_id, conn)
            # giving only the counter is not unique since we have 2 experiments
            with pytest.raises(NameError, match="More than one matching dataset"):
                load_by_run_spec(captured_counter=counter, conn=conn)
            # however we can supply counter and experiment
            ds2 = load_by_run_spec(
                captured_counter=counter, experiment_name="some-exp", conn=conn
            )

            assert isinstance(ds1, DataSet)
            assert ds1.the_same_dataset_as(ds2)
            assert ds1.counter == counter
            assert ds1.counter == ds1.captured_counter
            assert ds2.counter == counter
            assert ds2.counter == ds2.captured_counter


@pytest.mark.parametrize("db_file", ["empty", "some_runs"])
def test_perform_actual_upgrade_7_to_8(db_file) -> None:
    v7fixpath = os.path.join(fixturepath, "db_files", "version7")

    db_file += ".db"
    dbname_old = os.path.join(v7fixpath, db_file)

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=7) as conn:
        perform_db_upgrade_7_to_8(conn)

        assert is_column_in_table(conn, "runs", "parent_datasets")


@pytest.mark.usefixtures("empty_temp_db")
def test_cannot_connect_to_newer_db() -> None:
    conn = connect(qc.config["core"]["db_location"], qc.config["core"]["db_debug"])
    current_version = get_user_version(conn)
    set_user_version(conn, current_version + 1)
    conn.close()
    err_msg = (
        f"is version {current_version + 1} but this version of QCoDeS "
        f"supports up to version {current_version}"
    )
    with pytest.raises(RuntimeError, match=err_msg):
        conn = connect(qc.config["core"]["db_location"], qc.config["core"]["db_debug"])


def test_latest_available_version() -> None:
    assert _latest_available_version() == 9


@pytest.mark.parametrize("version", VERSIONS[:-1])
def test_getting_db_version(version) -> None:
    fixpath = os.path.join(fixturepath, "db_files", f"version{version}")

    dbname = os.path.join(fixpath, "empty.db")

    skip_if_no_fixtures(dbname)

    (db_v, new_v) = get_db_version_and_newest_available_version(dbname)

    assert db_v == version
    assert new_v == LATEST_VERSION


@pytest.mark.parametrize("db_file", ["empty", "some_runs"])
def test_perform_actual_upgrade_8_to_9(db_file) -> None:
    v8fixpath = os.path.join(fixturepath, "db_files", "version8")

    db_file += ".db"
    dbname_old = os.path.join(v8fixpath, db_file)

    skip_if_no_fixtures(dbname_old)

    with temporarily_copied_DB(dbname_old, debug=False, version=8) as conn:
        index_query = "PRAGMA index_list(runs)"

        c = atomic_transaction(conn, index_query)
        assert len(c.fetchall()) == 2

        perform_db_upgrade_8_to_9(conn)

        c = atomic_transaction(conn, index_query)
        assert len(c.fetchall()) == 3
