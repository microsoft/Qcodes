from hypothesis import given, settings
import hypothesis.strategies as hst
import numpy as np

import qcodes as qc
from qcodes import ParamSpec, new_data_set, new_experiment, experiments
from qcodes import load_by_id, load_by_counter
from qcodes.dataset.sqlite_base import connect, init_db, _unicode_categories
import qcodes.dataset.data_set
from qcodes.dataset.sqlite_base import get_user_version, set_user_version, atomic_transaction
from qcodes.dataset.data_set import CompletedError
from qcodes.dataset.database import initialise_database

import qcodes.dataset.experiment_container
import pytest
import tempfile
import os


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
    e = new_experiment("test-experiment", sample_name="test-sample")
    yield e
    e.conn.close()


@pytest.fixture(scope='function')
def dataset(experiment):
    dataset = new_data_set("test-dataset")
    yield dataset
    dataset.conn.close()


def test_load_by_id(experiment):
    ds = new_data_set("test-dataset")
    run_id = ds.run_id
    ds.mark_complete()

    loaded_ds = load_by_id(run_id)
    assert loaded_ds.completed == True
    assert loaded_ds.exp_id == 1

    ds = new_data_set("test-dataset-unfinished")
    run_id = ds.run_id

    loaded_ds = load_by_id(run_id)
    assert loaded_ds.completed == False
    assert loaded_ds.exp_id == 1