# Since all other tests of data_set and measurements will inevitably also
# test the sqlite_base module, we mainly test exceptions and small helper
# functions here
from sqlite3 import OperationalError
import tempfile
import os
from contextlib import contextmanager

import pytest
import hypothesis.strategies as hst
from hypothesis import given
import unicodedata
import numpy as np
from unittest.mock import patch

from qcodes.dataset.descriptions import RunDescriber
from qcodes.dataset.dependencies import InterDependencies
import qcodes.dataset.sqlite_base as mut  # mut: module under test
from qcodes.dataset.database import get_DB_location, path_to_dbfile
from qcodes.dataset.guids import generate_guid
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.param_spec import ParamSpec
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import \
    empty_temp_db, experiment, dataset
from qcodes.tests.dataset.dataset_fixtures import scalar_dataset, \
    standalone_parameters_dataset
from qcodes.tests.common import error_caused_by
# pylint: enable=unused-import

from .helper_functions import verify_data_dict

_unicode_categories = ('Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nd', 'Pc', 'Pd', 'Zs')


@contextmanager
def shadow_conn(path_to_db: str):
    """
    Simple context manager to create a connection for testing and
    close it on exit
    """
    conn = mut.connect(path_to_db)
    yield conn
    conn.close()


def test_path_to_dbfile():
    with tempfile.TemporaryDirectory() as tempdir:
        tempdb = os.path.join(tempdir, 'database.db')
        conn = mut.connect(tempdb)
        try:
            assert path_to_dbfile(conn) == tempdb
        finally:
            conn.close()


def test_one_raises(experiment):
    conn = experiment.conn

    with pytest.raises(RuntimeError):
        mut.one(conn.cursor(), column='Something_you_dont_have')


def test_atomic_transaction_raises(experiment):
    conn = experiment.conn

    bad_sql = '""'

    with pytest.raises(RuntimeError):
        mut.atomic_transaction(conn, bad_sql)


def test_atomic_raises(experiment):
    conn = experiment.conn

    bad_sql = '""'

    # it seems that the type of error raised differs between python versions
    # 3.6.0 (OperationalError) and 3.6.3 (RuntimeError)
    # -strange, huh?
    with pytest.raises((OperationalError, RuntimeError)):
        with mut.atomic(conn):
            mut.transaction(conn, bad_sql)


def test_insert_many_values_raises(experiment):
    conn = experiment.conn

    with pytest.raises(ValueError):
        mut.insert_many_values(conn, 'some_string', ['column1'],
                               values=[[1], [1, 3]])


def test_get_metadata_raises(experiment):
    with pytest.raises(RuntimeError) as excinfo:
        mut.get_metadata(experiment.conn, 'something', 'results')
    assert error_caused_by(excinfo, "no such column: something")


@given(table_name=hst.text(max_size=50))
def test__validate_table_raises(table_name):
    should_raise = False
    for char in table_name:
        if unicodedata.category(char) not in _unicode_categories:
            should_raise = True
            break
    if should_raise:
        with pytest.raises(RuntimeError):
            mut._validate_table_name(table_name)
    else:
        assert mut._validate_table_name(table_name)


def test_get_dependents(experiment):
    x = ParamSpec('x', 'numeric')
    t = ParamSpec('t', 'numeric')
    y = ParamSpec('y', 'numeric', depends_on=['x', 't'])

    # Make a dataset
    (_, run_id, _) = mut.create_run(experiment.conn,
                                    experiment.exp_id,
                                    name='testrun',
                                    guid=generate_guid(),
                                    parameters=[x, t, y])

    deps = mut.get_dependents(experiment.conn, run_id)

    layout_id = mut.get_layout_id(experiment.conn,
                                  'y', run_id)

    assert deps == [layout_id]

    # more parameters, more complicated dependencies

    x_raw = ParamSpec('x_raw', 'numeric')
    x_cooked = ParamSpec('x_cooked', 'numeric', inferred_from=['x_raw'])
    z = ParamSpec('z', 'numeric', depends_on=['x_cooked'])

    (_, run_id, _) = mut.create_run(experiment.conn,
                                    experiment.exp_id,
                                    name='testrun',
                                    guid=generate_guid(),
                                    parameters=[x, t, x_raw,
                                                x_cooked, y, z])

    deps = mut.get_dependents(experiment.conn, run_id)

    expected_deps = [mut.get_layout_id(experiment.conn, 'y', run_id),
                     mut.get_layout_id(experiment.conn, 'z', run_id)]

    assert deps == expected_deps


def test_column_in_table(dataset):
    assert mut.is_column_in_table(dataset.conn, "runs", "run_id")
    assert not mut.is_column_in_table(dataset.conn, "runs",
                                      "non-existing-column")


def test_run_exist(dataset):
    assert mut.run_exists(dataset.conn, dataset.run_id)
    assert not mut.run_exists(dataset.conn, dataset.run_id + 1)


def test_get_last_run(dataset):
    assert dataset.run_id == mut.get_last_run(dataset.conn, dataset.exp_id)


def test_get_last_run_no_runs(experiment):
    assert None is mut.get_last_run(experiment.conn, experiment.exp_id)


def test_get_last_experiment(experiment):
    assert experiment.exp_id == mut.get_last_experiment(experiment.conn)


def test_get_last_experiment_no_experiments(empty_temp_db):
    conn = mut.connect(get_DB_location())
    assert None is mut.get_last_experiment(conn)


def test_update_runs_description(dataset):
    invalid_descs = ['{}', 'description']

    for idesc in invalid_descs:
        with pytest.raises(ValueError):
            mut.update_run_description(dataset.conn, dataset.run_id, idesc)

    desc = RunDescriber(InterDependencies()).to_json()
    mut.update_run_description(dataset.conn, dataset.run_id, desc)


def test_runs_table_columns(empty_temp_db):
    """
    Ensure that the column names of a pristine runs table are what we expect
    """
    colnames = mut.RUNS_TABLE_COLUMNS.copy()
    conn = mut.connect(get_DB_location())
    query = "PRAGMA table_info(runs)"
    cursor = conn.cursor()
    for row in cursor.execute(query):
        colnames.remove(row['name'])

    assert colnames == []


@pytest.mark.filterwarnings("ignore:get_data")
def test_get_data_no_columns(scalar_dataset):
    ds = scalar_dataset
    ref = mut.get_data(ds.conn, ds.table_name, [])
    assert ref == [[]]


def test_get_parameter_data(scalar_dataset):
    ds = scalar_dataset
    input_names = ['param_3']

    data = mut.get_parameter_data(ds.conn, ds.table_name, input_names)

    assert len(data.keys()) == len(input_names)

    expected_names = {}
    expected_names['param_3'] = ['param_0', 'param_1', 'param_2',
                                 'param_3']
    expected_shapes = {}
    expected_shapes['param_3'] = [(10 ** 3,)] * 4

    expected_values = {}
    expected_values['param_3'] = [np.arange(10000 * a, 10000 * a + 1000)
                                  for a in range(4)]
    verify_data_dict(data, None, input_names, expected_names,
                     expected_shapes, expected_values)


def test_get_parameter_data_independent_parameters(
        standalone_parameters_dataset):
    ds = standalone_parameters_dataset
    params = mut.get_non_dependencies(ds.conn,
                                      ds.run_id)
    expected_toplevel_params = ['param_1', 'param_2', 'param_3']
    assert params == expected_toplevel_params

    data = mut.get_parameter_data(ds.conn, ds.table_name)

    assert len(data.keys()) == len(expected_toplevel_params)

    expected_names = {}
    expected_names['param_1'] = ['param_1']
    expected_names['param_2'] = ['param_2']
    expected_names['param_3'] = ['param_3', 'param_0']

    expected_shapes = {}
    expected_shapes['param_1'] = [(10 ** 3,)]
    expected_shapes['param_2'] = [(10 ** 3,)]
    expected_shapes['param_3'] = [(10 ** 3,)] * 2

    expected_values = {}
    expected_values['param_1'] = [np.arange(10000, 10000 + 1000)]
    expected_values['param_2'] = [np.arange(20000, 20000 + 1000)]
    expected_values['param_3'] = [np.arange(30000, 30000 + 1000),
                                  np.arange(0, 1000)]

    verify_data_dict(data, None, expected_toplevel_params, expected_names,
                     expected_shapes, expected_values)


def test_is_run_id_in_db(empty_temp_db):
    conn = mut.connect(get_DB_location())
    mut.new_experiment(conn, 'test_exp', 'no_sample')

    for _ in range(5):
        ds = DataSet(conn=conn, run_id=None)

    # there should now be run_ids 1, 2, 3, 4, 5 in the database
    good_ids = [1, 2, 3, 4, 5]
    try_ids = [1, 3, 9999, 23, 0, 1, 1, 3, 34]

    sorted_try_ids = np.unique(try_ids)

    expected_dict = {tid: (tid in good_ids) for tid in sorted_try_ids}

    acquired_dict = mut.is_run_id_in_database(conn, *try_ids)

    assert expected_dict == acquired_dict


def test_atomic_creation(experiment):
    """"
    Test that dataset creation is atomic. Test for
    https://github.com/QCoDeS/Qcodes/issues/1444
    """

    def just_throw(*args):
        raise RuntimeError("This breaks adding metadata")

    # first we patch add_meta_data to throw an exception
    # if create_data is not atomic this would create a partial
    # run in the db. Causing the next create_run to fail
    with patch('qcodes.dataset.sqlite_base.add_meta_data', new=just_throw):
        x = ParamSpec('x', 'numeric')
        t = ParamSpec('t', 'numeric')
        y = ParamSpec('y', 'numeric', depends_on=['x', 't'])
        with pytest.raises(RuntimeError,
                           match="Rolling back due to unhandled exception")as e:
            mut.create_run(experiment.conn,
                           experiment.exp_id,
                           name='testrun',
                           guid=generate_guid(),
                           parameters=[x, t,
                                       y],
                           metadata={'a': 1})
    assert error_caused_by(e, "This breaks adding metadata")
    # since we are starting from an empty database and the above transaction
    # should be rolled back there should be no runs in the run table
    runs = mut.transaction(experiment.conn,
                           'SELECT run_id FROM runs').fetchall()
    assert len(runs) == 0
    with shadow_conn(experiment.path_to_db) as new_conn:
        runs = mut.transaction(new_conn,
                               'SELECT run_id FROM runs').fetchall()
        assert len(runs) == 0

    # if the above was not correctly rolled back we
    # expect the next creation of a run to fail
    mut.create_run(experiment.conn,
                   experiment.exp_id,
                   name='testrun',
                   guid=generate_guid(),
                   parameters=[x, t,
                               y],
                   metadata={'a': 1})

    runs = mut.transaction(experiment.conn,
                           'SELECT run_id FROM runs').fetchall()
    assert len(runs) == 1

    with shadow_conn(experiment.path_to_db) as new_conn:
        runs = mut.transaction(new_conn,
                               'SELECT run_id FROM runs').fetchall()
        assert len(runs) == 1
