# Since all other tests of data_set and measurements will inevitably also
# test the sqlite module, we mainly test exceptions and small helper
# functions here
import logging
import re
import time
import unicodedata
from contextlib import contextmanager
from unittest.mock import patch

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import given
from pytest import LogCaptureFixture

import qcodes.dataset.descriptions.versioning.serialization as serial
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.experiment_container import load_or_create_experiment
from qcodes.dataset.guids import generate_guid

# mut: module under test
from qcodes.dataset.sqlite import connection as mut_conn
from qcodes.dataset.sqlite import database as mut_db
from qcodes.dataset.sqlite import queries as mut_queries
from qcodes.dataset.sqlite import query_helpers as mut_help
from qcodes.dataset.sqlite.connection import atomic_transaction, path_to_dbfile
from qcodes.dataset.sqlite.database import get_DB_location
from tests.common import error_caused_by

from .helper_functions import verify_data_dict

_unicode_categories = ("Lu", "Ll", "Lt", "Lm", "Lo", "Nd", "Pc", "Pd", "Zs")


@contextmanager
def shadow_conn(path_to_db: str):
    """
    Simple context manager to create a connection for testing and
    close it on exit
    """
    conn = mut_db.connect(path_to_db)
    yield conn
    conn.close()


@pytest.fixture(name="simple_run_describer")
def _make_simple_run_describer():
    x = ParamSpecBase("x", "numeric")
    t = ParamSpecBase("t", "numeric")
    y = ParamSpecBase("y", "numeric")

    paramtree = {y: (x, t)}

    interdependencies = InterDependencies_(dependencies=paramtree)
    rundescriber = RunDescriber(interdependencies)
    yield rundescriber


def test_path_to_dbfile(tmp_path) -> None:
    tempdb = str(tmp_path / "database.db")
    conn = mut_db.connect(tempdb)
    try:
        assert path_to_dbfile(conn) == tempdb
        assert conn.path_to_dbfile == tempdb
    finally:
        conn.close()


def test_one_raises_on_no_results(experiment) -> None:
    conn = experiment.conn

    with pytest.raises(RuntimeError, match="Expected one row"):
        mut_help.one(conn.cursor(), column="Something_you_dont_have")


def test_one_raises_on_more_than_one_result(experiment) -> None:
    conn = experiment.conn

    # create another experiment with so that the query below returns
    # MORE THAN ONE experiment id
    load_or_create_experiment(experiment.name + "2", experiment.sample_name)

    query = """
    SELECT exp_id
    FROM experiments
    """
    cur = atomic_transaction(conn, query)

    with pytest.raises(RuntimeError, match="Expected only one row"):
        mut_help.one(cur, column="exp_id")


def test_one_raises_on_wrong_column_name(experiment) -> None:
    conn = experiment.conn

    query = """
    SELECT exp_id
    FROM experiments
    """
    cur = atomic_transaction(conn, query)

    with pytest.raises(RuntimeError, match=re.escape("no such column: eXP_id")):
        mut_help.one(cur, column="eXP_id")


def test_one_raises_on_wrong_column_index(experiment) -> None:
    conn = experiment.conn

    query = """
    SELECT exp_id
    FROM experiments
    """
    cur = atomic_transaction(conn, query)

    with pytest.raises(IndexError):
        mut_help.one(cur, column=1)


def test_one_works_if_given_column_index(experiment) -> None:
    # This test relies on the fact that there's only one experiment in the
    # given database
    conn = experiment.conn

    query = """
    SELECT exp_id
    FROM experiments
    """
    cur = atomic_transaction(conn, query)

    exp_id = mut_help.one(cur, column=0)

    assert exp_id == experiment.exp_id


def test_one_works_if_given_column_name(experiment) -> None:
    # This test relies on the fact that there's only one experiment in the
    # given database
    conn = experiment.conn

    query = """
    SELECT exp_id
    FROM experiments
    """
    cur = atomic_transaction(conn, query)

    exp_id = mut_help.one(cur, column="exp_id")

    assert exp_id == experiment.exp_id


def test_atomic_transaction_raises(experiment) -> None:
    conn = experiment.conn

    bad_sql = '""'

    with pytest.raises(RuntimeError):
        mut_conn.atomic_transaction(conn, bad_sql)


def test_atomic_raises(experiment) -> None:
    conn = experiment.conn

    bad_sql = '""'

    with pytest.raises(RuntimeError) as excinfo:
        with mut_conn.atomic(conn):
            mut_conn.transaction(conn, bad_sql)
    assert error_caused_by(excinfo, "syntax error")


def test_insert_many_values_raises(experiment) -> None:
    conn = experiment.conn

    with pytest.raises(ValueError):
        mut_help.insert_many_values(
            conn, "some_string", ["column1"], values=[[1], [1, 3]]
        )


def test_get_non_existing_metadata_returns_none(experiment) -> None:
    assert (
        mut_queries.get_data_by_tag_and_table_name(
            experiment.conn, "something", "results"
        )
        is None
    )


@given(table_name=hst.text(max_size=50))
def test__validate_table_raises(table_name) -> None:
    should_raise = False
    for char in table_name:
        if unicodedata.category(char) not in _unicode_categories:
            should_raise = True
            break
    if should_raise:
        with pytest.raises(RuntimeError):
            mut_queries._validate_table_name(table_name)
    else:
        assert mut_queries._validate_table_name(table_name)


def test_get_dependents_simple(experiment, simple_run_describer) -> None:
    (_, run_id, _) = mut_queries.create_run(
        experiment.conn,
        experiment.exp_id,
        name="testrun",
        guid=generate_guid(),
        description=simple_run_describer,
    )

    deps = mut_queries._get_dependents(experiment.conn, run_id)

    layout_id = mut_queries._get_layout_id(experiment.conn, "y", run_id)

    assert deps == [layout_id]


def test_get_dependents(experiment) -> None:
    # more parameters, more complicated dependencies
    x = ParamSpecBase("x", "numeric")
    t = ParamSpecBase("t", "numeric")
    y = ParamSpecBase("y", "numeric")

    x_raw = ParamSpecBase("x_raw", "numeric")
    x_cooked = ParamSpecBase("x_cooked", "numeric")
    z = ParamSpecBase("z", "numeric")

    deps_param_tree = {y: (x, t), z: (x_cooked,)}
    inferred_param_tree: dict[ParamSpecBase, tuple[ParamSpecBase, ...]] = {
        x_cooked: (x_raw,)
    }
    interdeps = InterDependencies_(
        dependencies=deps_param_tree, inferences=inferred_param_tree
    )
    description = RunDescriber(interdeps=interdeps)
    (_, run_id, _) = mut_queries.create_run(
        experiment.conn,
        experiment.exp_id,
        name="testrun",
        guid=generate_guid(),
        description=description,
    )

    deps = mut_queries._get_dependents(experiment.conn, run_id)

    expected_deps = [
        mut_queries._get_layout_id(experiment.conn, "y", run_id),
        mut_queries._get_layout_id(experiment.conn, "z", run_id),
    ]

    assert deps == expected_deps


def test_column_in_table(dataset) -> None:
    assert mut_help.is_column_in_table(dataset.conn, "runs", "run_id")
    assert not mut_help.is_column_in_table(dataset.conn, "runs", "non-existing-column")


def test_run_exist(dataset) -> None:
    assert mut_queries.run_exists(dataset.conn, dataset.run_id)
    assert not mut_queries.run_exists(dataset.conn, dataset.run_id + 1)


def test_get_last_run(dataset) -> None:
    assert dataset.run_id == mut_queries.get_last_run(dataset.conn, dataset.exp_id)
    assert dataset.run_id == mut_queries.get_last_run(dataset.conn)


def test_get_last_run_no_runs(experiment) -> None:
    assert None is mut_queries.get_last_run(experiment.conn, experiment.exp_id)
    assert None is mut_queries.get_last_run(experiment.conn)


def test_get_last_experiment(experiment) -> None:
    assert experiment.exp_id == mut_queries.get_last_experiment(experiment.conn)


def test_get_last_experiment_no_experiments(empty_temp_db) -> None:
    conn = mut_db.connect(get_DB_location())
    assert None is mut_queries.get_last_experiment(conn)


def test_update_runs_description(dataset) -> None:
    invalid_descs = ["{}", "description"]

    for idesc in invalid_descs:
        with pytest.raises(ValueError):
            mut_queries.update_run_description(dataset.conn, dataset.run_id, idesc)

    desc = serial.to_json_for_storage(RunDescriber(InterDependencies_()))
    mut_queries.update_run_description(dataset.conn, dataset.run_id, desc)


def test_runs_table_columns(empty_temp_db) -> None:
    """
    Ensure that the column names of a pristine runs table are what we expect
    """
    colnames = list(mut_queries.RUNS_TABLE_COLUMNS)
    conn = mut_db.connect(get_DB_location())
    query = "PRAGMA table_info(runs)"
    cursor = conn.execute(query)
    description = mut_help.get_description_map(cursor)
    for row in cursor.fetchall():
        colnames.remove(row[description["name"]])

    assert colnames == []


def test_get_parameter_data(scalar_dataset) -> None:
    ds = scalar_dataset
    input_names = ["param_3"]

    data = mut_queries.get_parameter_data(ds.conn, ds.table_name, input_names)

    assert len(data.keys()) == len(input_names)

    expected_names = {"param_3": ["param_0", "param_1", "param_2", "param_3"]}
    expected_shapes = {"param_3": [(10**3,)] * 4}

    expected_values = {
        "param_3": [np.arange(10000 * a, 10000 * a + 1000) for a in range(4)]
    }
    verify_data_dict(
        data, None, input_names, expected_names, expected_shapes, expected_values
    )


def test_get_parameter_data_independent_parameters(
    standalone_parameters_dataset,
) -> None:
    ds = standalone_parameters_dataset

    paramspecs = ds.description.interdeps.non_dependencies
    params = [ps.name for ps in paramspecs]
    expected_toplevel_params = ["param_1", "param_2", "param_3"]
    assert params == expected_toplevel_params

    data = mut_queries.get_parameter_data(ds.conn, ds.table_name)

    assert len(data.keys()) == len(expected_toplevel_params)

    expected_names = {
        "param_1": ["param_1"],
        "param_2": ["param_2"],
        "param_3": ["param_3", "param_0"],
    }

    expected_shapes = {
        "param_1": [(10**3,)],
        "param_2": [(10**3,)],
        "param_3": [(10**3,)] * 2,
    }
    expected_values = {
        "param_1": [np.arange(10000, 10000 + 1000)],
        "param_2": [np.arange(20000, 20000 + 1000)],
        "param_3": [np.arange(30000, 30000 + 1000), np.arange(0, 1000)],
    }

    verify_data_dict(
        data,
        None,
        expected_toplevel_params,
        expected_names,
        expected_shapes,
        expected_values,
    )


def test_is_run_id_in_db(empty_temp_db) -> None:
    conn = mut_db.connect(get_DB_location())
    mut_queries.new_experiment(conn, "test_exp", "no_sample")

    for _ in range(5):
        DataSet(conn=conn, run_id=None)

    # there should now be run_ids 1, 2, 3, 4, 5 in the database
    good_ids = [1, 2, 3, 4, 5]
    try_ids = [1, 3, 9999, 23, 0, 1, 1, 3, 34]

    sorted_try_ids = np.unique(try_ids)

    expected_dict = {tid: (tid in good_ids) for tid in sorted_try_ids}

    acquired_dict = mut_queries.is_run_id_in_database(conn, *try_ids)

    assert expected_dict == acquired_dict


def test_atomic_creation(experiment, simple_run_describer) -> None:
    """ "
    Test that dataset creation is atomic. Test for
    https://github.com/QCoDeS/Qcodes/issues/1444
    """

    def just_throw(*args):
        raise RuntimeError("This breaks adding metadata")

    # first we patch add_data_to_dynamic_columns to throw an exception
    # if create_data is not atomic this would create a partial
    # run in the db. Causing the next create_run to fail
    with patch(
        "qcodes.dataset.sqlite.queries.add_data_to_dynamic_columns", new=just_throw
    ):
        with pytest.raises(
            RuntimeError, match="Rolling back due to unhandled exception"
        ) as e:
            mut_queries.create_run(
                experiment.conn,
                experiment.exp_id,
                name="testrun",
                guid=generate_guid(),
                description=simple_run_describer,
                metadata={"a": 1},
            )
    assert error_caused_by(e, "This breaks adding metadata")
    # since we are starting from an empty database and the above transaction
    # should be rolled back there should be no runs in the run table
    runs = mut_conn.transaction(experiment.conn, "SELECT run_id FROM runs").fetchall()
    assert len(runs) == 0
    with shadow_conn(experiment.path_to_db) as new_conn:
        runs = mut_conn.transaction(new_conn, "SELECT run_id FROM runs").fetchall()
        assert len(runs) == 0

    # if the above was not correctly rolled back we
    # expect the next creation of a run to fail
    mut_queries.create_run(
        experiment.conn,
        experiment.exp_id,
        name="testrun",
        guid=generate_guid(),
        description=simple_run_describer,
        metadata={"a": 1},
    )

    runs = mut_conn.transaction(experiment.conn, "SELECT run_id FROM runs").fetchall()
    assert len(runs) == 1

    with shadow_conn(experiment.path_to_db) as new_conn:
        runs = mut_conn.transaction(new_conn, "SELECT run_id FROM runs").fetchall()
        assert len(runs) == 1


def test_set_run_timestamp(dataset) -> None:
    assert dataset.run_timestamp_raw is None
    assert dataset.completed_timestamp_raw is None

    time_now = time.time()
    time.sleep(1)  # for slower test platforms
    mut_queries.set_run_timestamp(dataset.conn, dataset.run_id)

    assert dataset.run_timestamp_raw is not None
    assert dataset.run_timestamp_raw > time_now
    assert dataset.completed_timestamp_raw is None

    with pytest.raises(
        RuntimeError, match="Rolling back due to unhandled exception"
    ) as ei:
        mut_queries.set_run_timestamp(dataset.conn, dataset.run_id)

    assert error_caused_by(ei, ("Can not set run_timestamp; it has already been set"))


def test_set_run_timestamp_explicit(dataset) -> None:
    assert dataset.run_timestamp_raw is None
    assert dataset.completed_timestamp_raw is None

    time_now = time.time()
    time.sleep(1)  # for slower test platforms
    mut_queries.set_run_timestamp(dataset.conn, dataset.run_id, time_now)

    assert dataset.run_timestamp_raw == time_now
    assert dataset.completed_timestamp_raw is None

    with pytest.raises(
        RuntimeError, match="Rolling back due to unhandled exception"
    ) as ei:
        mut_queries.set_run_timestamp(dataset.conn, dataset.run_id)

    assert error_caused_by(ei, "Can not set run_timestamp; it has already been set")


def test_mark_run_complete(dataset) -> None:
    assert dataset.run_timestamp_raw is None
    assert dataset.completed_timestamp_raw is None

    time_now = time.time()
    mut_queries.set_run_timestamp(dataset.conn, dataset.run_id)
    assert dataset.run_timestamp_raw is not None
    assert dataset.completed_timestamp_raw is None
    time.sleep(1)  # for slower test platforms
    mut_queries.mark_run_complete(dataset.conn, dataset.run_id)
    assert dataset.run_timestamp_raw is not None
    assert dataset.completed_timestamp_raw is not None
    assert dataset.completed_timestamp_raw > time_now


def test_mark_run_complete_twice(dataset, caplog: LogCaptureFixture) -> None:
    assert dataset.run_timestamp_raw is None
    assert dataset.completed_timestamp_raw is None

    time_now = time.time()
    mut_queries.set_run_timestamp(dataset.conn, dataset.run_id)
    assert dataset.run_timestamp_raw is not None
    assert dataset.completed_timestamp_raw is None
    time.sleep(1)  # for slower test platforms
    mut_queries.mark_run_complete(dataset.conn, dataset.run_id)
    assert dataset.run_timestamp_raw is not None
    assert dataset.completed_timestamp_raw is not None
    completed_time = dataset.completed_timestamp_raw
    assert completed_time > time_now

    # now wait a sec and mark the run complted again
    # this should not update the completed time
    # since the run is already complted
    time.sleep(1)  # for slower test platforms
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        mut_queries.mark_run_complete(dataset.conn, dataset.run_id)
    assert (
        caplog.records[0].msg
        == "Trying to mark a run completed that was already completed."
    )
    assert caplog.records[0].levelname == "WARNING"
    assert dataset.run_timestamp_raw is not None
    assert dataset.completed_timestamp_raw is not None
    assert dataset.completed_timestamp_raw == completed_time

    # however we can force an update with override
    mut_queries.mark_run_complete(dataset.conn, dataset.run_id, override=True)
    assert dataset.run_timestamp_raw is not None
    assert dataset.completed_timestamp_raw is not None
    assert dataset.completed_timestamp_raw > completed_time


def test_mark_run_complete_explicit_time(dataset) -> None:
    assert dataset.run_timestamp_raw is None
    assert dataset.completed_timestamp_raw is None

    mut_queries.set_run_timestamp(dataset.conn, dataset.run_id)
    time_now = time.time()
    time.sleep(1)  # for slower test platforms
    mut_queries.mark_run_complete(dataset.conn, dataset.run_id, time_now)

    assert dataset.completed_timestamp_raw == time_now

    mut_queries.mark_run_complete(dataset.conn, dataset.run_id)
