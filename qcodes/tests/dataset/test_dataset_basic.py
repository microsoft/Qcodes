import qcodes as qc
from qcodes import ParamSpec, new_data_set, new_experiment, experiments, load_by_id
from qcodes.dataset.sqlite_base import connect, init_db, connect
import qcodes.dataset.data_set
import pytest
import tempfile
import os


@pytest.fixture(scope="module")
def setup_temp_database():
    # create a temp database for testing
    print("setting up db")
    with tempfile.TemporaryDirectory() as tmpdirname:
        # todo make sure this is safely deleted
        qc.config["core"]["db_location"] = os.path.join(tmpdirname, 'temp.db')
        qc.config["core"]["db_debug"] = True
        qc.dataset.experiment_container.DB = qc.config["core"]["db_location"]
        qc.dataset.data_set.DB = qcodes.config["core"]["db_location"]
        qc.dataset.experiment_container.debug_db = qc.config["core"]["db_debug"]
        _c = connect(qc.config["core"]["db_location"], qc.config["core"]["db_debug"])
        init_db(_c)
        _c.close()
        yield



def test_tabels_exists(setup_temp_database):
    print(qc.config["core"]["db_location"])
    conn = connect(qc.config["core"]["db_location"], qc.config["core"]["db_debug"])
    cursor = conn.execute("select sql from sqlite_master where type = 'table'")
    expected_tables = ['experiments', 'runs', 'layouts', 'dependencies']
    for row, expected_table in zip(cursor, expected_tables):
        assert expected_table in row['sql']



def test_add_experiment(setup_temp_database):
    print(qc.config["core"]["db_location"])
    print(qc.dataset.experiment_container.DB)
    e = new_experiment("testing-experiment", sample_name="testingsample")
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "testing-experiment"
    assert exp.sample_name == "testingsample"
    assert exp.last_counter == 0
    dataset = new_data_set("sweep gate")
    dsid = dataset.id
    loaded_dataset = load_by_id(dsid)
    assert loaded_dataset.name == "sweep gate"
    assert loaded_dataset.counter == 1
    assert loaded_dataset.table_name == "{}-{}-{}".format(loaded_dataset.name,
                                                          exp.id,
                                                          loaded_dataset.counter)
    dataset = new_data_set("sweep gate")
    dsid = dataset.id
    loaded_dataset = load_by_id(dsid)
    assert loaded_dataset.name == "sweep gate"
    assert loaded_dataset.counter == 2
    assert loaded_dataset.table_name == "{}-{}-{}".format(loaded_dataset.name,
                                                          exp.id,
                                                          loaded_dataset.counter)
    parameter_a = ParamSpec("a", "INTEGER")
    parameter_b = ParamSpec("b", "INTEGER", key="value", number=1)
    parameter_c = ParamSpec("c", "array")
#
