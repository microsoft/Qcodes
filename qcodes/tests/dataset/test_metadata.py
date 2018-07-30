import os
import sqlite3
import tempfile

import pytest

import qcodes as qc
from qcodes import new_data_set
from qcodes.dataset.database import initialise_database


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
    e = qc.new_experiment('experiment_name', 'sample_name')
    yield e
    e.conn.close()


@pytest.fixture(scope='function')
def dataset(experiment):
    dataset = new_data_set("dataset_name")
    yield dataset
    dataset.conn.close()


def test_get_metadata_from_dataset(dataset):
    dataset.add_metadata('something', 123)
    something = dataset.get_metadata('something')
    assert 123 == something


def test_get_nonexisting_metadata(dataset):
    with pytest.raises(sqlite3.OperationalError,
                       match="no such column: something"):
        _ = dataset.get_metadata('something')
