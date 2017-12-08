import qcodes as qc
from qcodes import ParamSpec, new_data_set, new_experiment, experiments, load_by_id
from qcodes.dataset.sqlite_base import connect, init_db, connect, _unicode_categories
import qcodes.dataset.data_set
import qcodes.dataset.experiment_container
import pytest
import tempfile
import os

from hypothesis import given, settings
import hypothesis.strategies as hst

n_experiments = 0

@pytest.fixture(scope="function")
def empty_temp_db():
    # create a temp database for testing
    print("setting up db")
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
def temp_db_with_exp_and_ds(empty_temp_db):
    e = new_experiment("test-experiment", sample_name="test-sample")
    #dataset = new_data_set("test-dataset")

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
    e = new_experiment(experiment_name, sample_name=sample_name)
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


def test_add_paramspec(temp_db_with_exp_and_ds):
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 0
