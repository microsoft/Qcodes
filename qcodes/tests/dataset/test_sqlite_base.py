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

_unicode_categories = ('Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nd', 'Pc', 'Pd', 'Zs')


@pytest.fixture(scope="function")
def empty_temp_db():
    # create a temp database for testing
    with tempfile.TemporaryDirectory() as tmpdirname:
        qc.config["core"]["db_location"] = os.path.join(tmpdirname, 'temp.db')
        qc.config["core"]["db_debug"] = False
        # this is somewhat annoying but these module scope variables
        # are initialized at import time so they need to be overwritten
        qc.dataset.experiment_container.DB = qc.config["core"]["db_location"]
        qc.dataset.data_set.DB = qc.config["core"]["db_location"]
        qc.dataset.experiment_container.debug_db = qc.config["core"]["db_debug"]
        _c = mut.connect(qc.config["core"]["db_location"],
                         qc.config["core"]["db_debug"])
        mut.init_db(_c)
        _c.close()
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
