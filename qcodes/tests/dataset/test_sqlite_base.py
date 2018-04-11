# Since all other tests of data_set and measurements will inevitably also
# test the sqlite_base module, we mainly test exceptions here

import tempfile
import os
from sqlite3 import OperationalError

import pytest
import hypothesis.strategies as hst
from hypothesis import given
import unicodedata

import qcodes as qc
import qcodes.dataset.sqlite_base as mut  # mut: module under test
from qcodes.dataset.database import initialise_database
from qcodes.dataset.param_spec import ParamSpec

_unicode_categories = ('Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nd', 'Pc', 'Pd', 'Zs')


@pytest.fixture(scope="function")
def empty_temp_db():
    # create a temp database for testing
    with tempfile.TemporaryDirectory() as tmpdirname:
        qc.config["core"]["db_location"] = os.path.join(tmpdirname, 'temp.db')
        qc.config["core"]["db_debug"] = True
        initialise_database()
        yield


@pytest.fixture(scope='function')
def experiment(empty_temp_db):
    e = qc.new_experiment("test-experiment", sample_name="test-sample")
    yield e
    e.conn.close()


def test_one_raises(experiment):
    conn = experiment.conn

    with pytest.raises(RuntimeError):
        mut.one(conn.cursor(), column='Something_you_dont_have')


def test_atomicTransaction_raises(experiment):
    conn = experiment.conn

    bad_sql = '""'

    with pytest.raises(OperationalError):
        mut.atomicTransaction(conn, bad_sql)


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
    (rc, run_id, fmt_name) = mut.create_run(experiment.conn,
                                            experiment.exp_id,
                                            name='testrun',
                                            parameters=[x, t, y])

    deps = mut.get_dependents(experiment.conn, run_id)

    assert deps == [3]
