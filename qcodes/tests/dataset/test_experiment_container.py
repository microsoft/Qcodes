import pytest
import tempfile
import os

import qcodes as qc
from qcodes.dataset.experiment_container import load_experiment_by_name, \
    new_experiment, load_or_create_experiment, experiments
from qcodes.dataset.sqlite_base import connect, init_db
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.database import initialise_database


def assert_experiments_equal(exp, exp_2):
    for attr in ['name', 'sample_name', 'path_to_db', 'last_counter']:
        assert getattr(exp, attr) == getattr(exp_2, attr)
    assert len(exp_2) == len(exp)
    assert repr(exp_2) == repr(exp)


@pytest.fixture(scope="function")
def empty_temp_db():
    # create a temp database for testing
    with tempfile.TemporaryDirectory() as tmpdirname:
        qc.config["core"]["db_location"] = os.path.join(tmpdirname, 'temp.db')
        qc.config["core"]["db_debug"] = True
        initialise_database()
        yield


def test_run_loaded_experiment(empty_temp_db):
    """
    Test that we can resume a measurement after loading by name
    """
    new_experiment("test", "test1")
    exp_loaded = load_experiment_by_name("test", "test1")

    meas = Measurement(exp=exp_loaded)
    with meas.run():
        pass

    with meas.run():
        pass


def test_load_or_create_experiment_loading(empty_temp_db):
    """Test that an experiment is correctly loaded"""
    exp = new_experiment("experiment_name", "sample_name")
    exp_2 = load_or_create_experiment("experiment_name", "sample_name")
    assert_experiments_equal(exp, exp_2)


def test_load_or_create_experiment_different_sample_name(empty_temp_db):
    """
    Test that an experiment is created for the case when the experiment
    name is the same, but the sample name is different
    """
    exp = new_experiment("experiment_name", "sample_name_1")
    exp_2 = load_or_create_experiment("experiment_name", "sample_name_2")

    actual_experiments = experiments()
    assert len(actual_experiments) == 2

    assert exp.name == exp_2.name
    assert exp.sample_name != exp_2.sample_name


def test_load_or_create_experiment_creating(empty_temp_db):
    """Test that an experiment is correctly created"""
    exp = load_or_create_experiment("experiment_name", "sample_name")
    exp_2 = load_experiment_by_name("experiment_name", "sample_name")
    assert_experiments_equal(exp, exp_2)


def test_load_or_create_experiment_creating_not_empty(empty_temp_db):
    """Test that an experiment is correctly created when DB is not empty"""
    exp = load_or_create_experiment("experiment_name_1", "sample_name_1")
    exp_2 = load_or_create_experiment("experiment_name_2", "sample_name_2")

    actual_experiments = experiments()
    assert len(actual_experiments) == 2

    assert_experiments_equal(actual_experiments[0], exp)
    assert_experiments_equal(actual_experiments[1], exp_2)
