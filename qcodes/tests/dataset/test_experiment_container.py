import pytest
import tempfile
import os

import qcodes as qc
from qcodes.dataset.experiment_container import load_experiment_by_name, \
    new_experiment
from qcodes.dataset.sqlite_base import connect, init_db
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.database import initialise_database


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
