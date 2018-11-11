import tempfile
import os
from contextlib import contextmanager
import shutil

import pytest

import qcodes as qc
from qcodes.dataset.database import initialise_database
from qcodes import new_experiment, new_data_set
from qcodes.dataset.sqlite_base import connect

n_experiments = 0


@pytest.fixture(scope="function")
def empty_temp_db():
    global n_experiments
    n_experiments = 0
    # create a temp database for testing
    with tempfile.TemporaryDirectory() as tmpdirname:
        qc.config["core"]["db_location"] = os.path.join(tmpdirname, 'temp.db')
        qc.config["core"]["db_debug"] = True
        initialise_database()
        yield


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
