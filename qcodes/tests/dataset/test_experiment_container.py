import pytest

from qcodes.dataset.experiment_container import (load_experiment_by_name,
                                                 new_experiment,
                                                 load_or_create_experiment,
                                                 experiments,
                                                 Experiment)
from qcodes.dataset.measurements import Measurement
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import empty_temp_db


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


@pytest.mark.usefixtures("empty_temp_db")
def test_has_attributes_after_init():
    """
    Ensure that all attributes are populated after __init__ in BOTH cases
    (exp_id is None / exp_id is not None)
    """

    attrs = ['name', 'sample_name', 'last_counter', 'path_to_db',
             'conn', 'started_at', 'finished_at']

    # This creates a run in the db
    exp1 = Experiment(exp_id=None)

    # This loads the run that we just created
    exp2 = Experiment(exp_id=1)

    # This tries to load an experiment that we don't have
    with pytest.raises(ValueError, match='No such experiment in the database'):
        Experiment(exp_id=2)

    for exp in (exp1, exp2):
        for attr in attrs:
            assert hasattr(exp, attr)
            getattr(exp, attr)
