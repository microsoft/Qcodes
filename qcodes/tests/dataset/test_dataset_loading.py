import os

import pytest
import tempfile

import qcodes as qc
from qcodes.dataset.database import initialise_database
from qcodes.dataset.experiment_container import new_experiment
from qcodes.dataset.data_set import new_data_set, load_by_id, load_by_counter


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


def test_load_by_id(experiment):
    ds = new_data_set("test-dataset")
    run_id = ds.run_id
    ds.mark_complete()

    loaded_ds = load_by_id(run_id)
    assert loaded_ds.completed is True
    assert loaded_ds.exp_id == 1

    ds = new_data_set("test-dataset-unfinished")
    run_id = ds.run_id

    loaded_ds = load_by_id(run_id)
    assert loaded_ds.completed is False
    assert loaded_ds.exp_id == 1


def test_load_by_counter(empty_temp_db):
    exp = new_experiment(name="for_loading", sample_name="no_sample")
    ds = new_data_set("my_first_ds")

    loaded_ds = load_by_counter(exp.exp_id, 1)

    assert loaded_ds.completed is False

    ds.mark_complete()
    loaded_ds = load_by_counter(exp.exp_id, 1)

    assert loaded_ds.completed is True
