import tempfile
import os

import pytest

import qcodes as qc
from qcodes.dataset.database import initialise_database
from qcodes import new_experiment, new_data_set

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
    yield e
    e.conn.close()

@pytest.fixture(scope='function')
def dataset(experiment):
    dataset = new_data_set("test-dataset")
    yield dataset
    dataset.conn.close()
