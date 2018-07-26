import os
import tempfile
import time
from math import floor

import pytest

import qcodes as qc
from qcodes.dataset.data_set import new_data_set, load_by_id, load_by_counter
from qcodes.dataset.database import initialise_database
from qcodes.dataset.experiment_container import new_experiment


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


def test_experiment_info_in_dataset(empty_temp_db):
    exp = new_experiment(name="for_loading", sample_name="no_sample")
    ds = new_data_set("my_first_ds")

    assert ds.exp_id == exp.exp_id
    assert ds.exp_name == exp.name
    assert ds.sample_name == exp.sample_name


def test_run_timestamp(empty_temp_db):
    _ = new_experiment(name="for_loading", sample_name="no_sample")

    t_before_data_set = time.time()
    ds = new_data_set("my_first_ds")
    t_after_data_set = time.time()

    actual_run_timestamp_raw = ds.run_timestamp_raw

    assert t_before_data_set < actual_run_timestamp_raw < t_after_data_set


def test_run_timestamp_with_default_format(empty_temp_db):
    _ = new_experiment(name="for_loading", sample_name="no_sample")

    t_before_data_set = time.time()
    ds = new_data_set("my_first_ds")
    t_after_data_set = time.time()

    # Note that here we also test the default format of `run_timestamp`
    actual_run_timestamp_raw = time.mktime(
        time.strptime(ds.run_timestamp, "%Y-%m-%d %H:%M:%S"))

    # Note that because the default format precision is 1 second, we add this
    # second to the right side of the comparison
    t_before_data_set_secs = floor(t_before_data_set)
    t_after_data_set_secs = floor(t_after_data_set)
    assert t_before_data_set_secs \
           <= actual_run_timestamp_raw \
           <= t_after_data_set_secs + 1
