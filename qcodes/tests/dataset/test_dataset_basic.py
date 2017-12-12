import qcodes as qc
from qcodes import ParamSpec, new_data_set, new_experiment, experiments
from qcodes import load_by_id, load_by_counter
from qcodes.dataset.sqlite_base import connect, init_db, connect, _unicode_categories
import qcodes.dataset.data_set
import qcodes.dataset.experiment_container
import pytest
import tempfile
import os

from hypothesis import given
import hypothesis.strategies as hst
import numpy as np

n_experiments = 0


@pytest.fixture(scope="function")
def empty_temp_db():
    # create a temp database for testing
    with tempfile.TemporaryDirectory() as tmpdirname:
        qc.config["core"]["db_location"] = os.path.join(tmpdirname, 'temp.db')
        qc.config["core"]["db_debug"] = True
        # this is somewhat annoying but these module scope variables
        # are initialized at import time so they need to be overwritten
        qc.dataset.experiment_container.DB = qc.config["core"]["db_location"]
        qc.dataset.data_set.DB = qcodes.config["core"]["db_location"]
        qc.dataset.experiment_container.debug_db = qc.config["core"]["db_debug"]
        _c = connect(qc.config["core"]["db_location"], qc.config["core"]["db_debug"])
        init_db(_c)
        _c.close()
        yield


@pytest.fixture(scope='function')
def experiment(empty_temp_db):
    e = new_experiment("test-experiment", sample_name="test-sample")
    yield e


@pytest.fixture(scope='function')
def dataset(experiment):
    dataset = new_data_set("test-dataset")
    yield dataset


def test_tabels_exists(empty_temp_db):
    print(qc.config["core"]["db_location"])
    conn = connect(qc.config["core"]["db_location"], qc.config["core"]["db_debug"])
    cursor = conn.execute("select sql from sqlite_master where type = 'table'")
    expected_tables = ['experiments', 'runs', 'layouts', 'dependencies']
    for row, expected_table in zip(cursor, expected_tables):
        assert expected_table in row['sql']


@given(experiment_name=hst.text(min_size=1),
       sample_name=hst.text(min_size=1),
       dataset_name=hst.text(hst.characters(whitelist_categories=_unicode_categories),
                             min_size=1))
def test_add_experiments(empty_temp_db, experiment_name,
                         sample_name, dataset_name):
    global n_experiments
    n_experiments += 1

    _ = new_experiment(experiment_name, sample_name=sample_name)
    exps = experiments()
    assert len(exps) == n_experiments
    exp = exps[-1]
    assert exp.name == experiment_name
    assert exp.sample_name == sample_name
    assert exp.last_counter == 0

    dataset = new_data_set(dataset_name)
    dsid = dataset.id
    loaded_dataset = load_by_id(dsid)
    expected_ds_counter = 1
    assert loaded_dataset.name == dataset_name
    assert loaded_dataset.counter == expected_ds_counter
    assert loaded_dataset.table_name == "{}-{}-{}".format(dataset_name,
                                                          exp.id,
                                                          loaded_dataset.counter)
    expected_ds_counter += 1
    dataset = new_data_set(dataset_name)
    dsid = dataset.id
    loaded_dataset = load_by_id(dsid)
    assert loaded_dataset.name == dataset_name
    assert loaded_dataset.counter == expected_ds_counter
    assert loaded_dataset.table_name == "{}-{}-{}".format(dataset_name,
                                                          exp.id,
                                                          loaded_dataset.counter)


def test_add_paramspec(dataset):
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 1

    parameter_a = ParamSpec("a", "INTEGER")
    parameter_b = ParamSpec("b", "INTEGER", key="value", number=1)
    parameter_c = ParamSpec("c", "array")
    dataset.add_parameters([parameter_a, parameter_b, parameter_c])
    paramspecs = dataset.paramspecs
    expected_keys = ['a', 'b', 'c']
    keys = sorted(list(paramspecs.keys()))
    assert keys == expected_keys
    for expected_param_name in expected_keys:
        ps = paramspecs[expected_param_name]
        assert ps.name == expected_param_name


def test_add_paramspec_one_by_one(dataset):
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 1

    parameters = [ParamSpec("a", "INTEGER"),
                  ParamSpec("b", "INTEGER", key="value", number=1),
                  ParamSpec("c", "array")]
    for parameter in parameters:
        dataset.add_parameter(parameter)
    paramspecs = dataset.paramspecs
    expected_keys = ['a', 'b', 'c']
    keys = sorted(list(paramspecs.keys()))
    assert keys == expected_keys
    for expected_param_name in expected_keys:
        ps = paramspecs[expected_param_name]
        assert ps.name == expected_param_name

    with pytest.raises(ValueError):
        dataset.add_parameter(parameters[0])
    assert len(dataset.paramspecs.keys()) == 3


def test_add_data_1d(experiment):
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 0

    mydataset = new_data_set("test-dataset", specs=[ParamSpec("x", "number"),
                                                    ParamSpec("y", "number")])

    expected_x = []
    expected_y = []
    for x in range(100):
        expected_x.append([x])
        y = 3 * x + 10
        expected_y.append([y])
        mydataset.add_result({"x": x, "y": y})
    assert mydataset.get_data('x') == expected_x
    assert mydataset.get_data('y') == expected_y
    assert mydataset.completed is False
    mydataset.mark_complete()
    assert mydataset.completed is True


def test_add_data_array(experiment):
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 0

    mydataset = new_data_set("test", specs=[ParamSpec("x", "number"),
                                            ParamSpec("y", "array")])

    expected_x = []
    expected_y = []
    for x in range(100):
        expected_x.append([x])
        y = np.random.random_sample(10)
        expected_y.append([y])
        mydataset.add_result({"x": x, "y": y})
    assert mydataset.get_data('x') == expected_x
    y_data = mydataset.get_data('y')
    np.testing.assert_allclose(y_data, expected_y)


def test_load_by_counter(dataset):
    dataset = load_by_counter(1, 1)
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert dataset.name == "test-dataset"
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 1


def test_dataset_with_no_experiment_raises(empty_temp_db):
    with pytest.raises(ValueError):
        mydataset = new_data_set("test-dataset",
                                 specs=[ParamSpec("x", "number"),
                                        ParamSpec("y", "number")])
