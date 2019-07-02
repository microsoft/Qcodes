import tempfile
import os
from contextlib import contextmanager
import shutil

import pytest

import qcodes as qc
from qcodes.dataset.sqlite.database import initialise_database, connect
from qcodes import new_experiment, new_data_set


n_experiments = 0


@pytest.fixture(scope="function")
def empty_temp_db():
    global n_experiments
    n_experiments = 0
    # create a temp database for testing
    with tempfile.TemporaryDirectory() as tmpdirname:
        qc.config["core"]["db_location"] = os.path.join(tmpdirname, 'temp.db')
        if os.environ.get('QCODES_SQL_DEBUG'):
            qc.config["core"]["db_debug"] = True
        else:
            qc.config["core"]["db_debug"] = False
        initialise_database()
        yield


@pytest.fixture(scope='function')
def empty_temp_db_connection():
    """
    Yield connection to an empty temporary DB file.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, 'source.db')
        conn = connect(path)
        try:
            yield conn
        finally:
            conn.close()


@pytest.fixture(scope='function')
def two_empty_temp_db_connections():
    """
    Yield the paths of two empty files. Meant for use with the
    test_database_copy_paste
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        source_path = os.path.join(tmpdirname, 'source.db')
        target_path = os.path.join(tmpdirname, 'target.db')
        source_conn = connect(source_path)
        target_conn = connect(target_path)
        try:
            yield (source_conn, target_conn)
        finally:
            source_conn.close()
            target_conn.close()


@pytest.fixture(scope='function')
def experiment(empty_temp_db):
    e = new_experiment("test-experiment", sample_name="test-sample")
    try:
        yield e
    finally:
        e.conn.close()


@pytest.fixture(scope='function')
def dataset(experiment):
    dataset = new_data_set("test-dataset")
    try:
        yield dataset
    finally:
        dataset.unsubscribe_all()
        dataset.conn.close()


@contextmanager
def temporarily_copied_DB(filepath: str, **kwargs):
    """
    Make a temporary copy of a db-file and delete it after use. Meant to be
    used together with the old version database fixtures, lest we change the
    fixtures on disk. Yields the connection object

    Args:
        filepath: path to the db-file

    Kwargs:
        kwargs to be passed to connect
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dbname_new = os.path.join(tmpdir, 'temp.db')
        shutil.copy2(filepath, dbname_new)

        conn = connect(dbname_new, **kwargs)

        try:
            yield conn

        finally:
            conn.close()
