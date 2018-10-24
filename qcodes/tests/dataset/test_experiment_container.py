import pytest

from qcodes.dataset.experiment_container import load_experiment_by_name, \
    new_experiment, load_or_create_experiment, experiments, load_experiment
from qcodes.dataset.measurements import Measurement
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import empty_temp_db, dataset, \
    experiment


def assert_experiments_equal(exp, exp_2):
    for attr in ['name', 'sample_name', 'path_to_db', 'last_counter']:
        assert getattr(exp, attr) == getattr(exp_2, attr)
    assert len(exp_2) == len(exp)
    assert repr(exp_2) == repr(exp)


@pytest.mark.usefixtures("empty_temp_db")
def test_run_loaded_experiment():
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

def test_last_data_set_from_experiment(dataset):
    experiment = load_experiment(dataset.exp_id)
    ds = experiment.last_data_set()

    assert dataset.run_id == ds.run_id
    assert dataset.name == ds.name
    assert dataset.exp_id == ds.exp_id
    assert dataset.exp_name == ds.exp_name
    assert dataset.sample_name == ds.sample_name
    assert dataset.path_to_db == ds.path_to_db

    assert experiment.path_to_db == ds.path_to_db


def test_last_data_set_from_experiment_with_no_datasets(experiment):
    with pytest.raises(ValueError, match='There are no runs in this '
                                         'experiment'):
        _ = experiment.last_data_set()


@pytest.mark.usefixtures("empty_temp_db")
def test_load_or_create_experiment_loading():
    """Test that an experiment is correctly loaded"""
    exp = new_experiment("experiment_name", "sample_name")
    exp_2 = load_or_create_experiment("experiment_name", "sample_name")
    assert_experiments_equal(exp, exp_2)


@pytest.mark.usefixtures("empty_temp_db")
def test_load_or_create_experiment_different_sample_name():
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


@pytest.mark.usefixtures("empty_temp_db")
def test_load_or_create_experiment_creating():
    """Test that an experiment is correctly created"""
    exp = load_or_create_experiment("experiment_name", "sample_name")
    exp_2 = load_experiment_by_name("experiment_name", "sample_name")
    assert_experiments_equal(exp, exp_2)


@pytest.mark.usefixtures("empty_temp_db")
def test_load_or_create_experiment_creating_not_empty():
    """Test that an experiment is correctly created when DB is not empty"""
    exp = load_or_create_experiment("experiment_name_1", "sample_name_1")
    exp_2 = load_or_create_experiment("experiment_name_2", "sample_name_2")

    actual_experiments = experiments()
    assert len(actual_experiments) == 2

    assert_experiments_equal(actual_experiments[0], exp)
    assert_experiments_equal(actual_experiments[1], exp_2)
