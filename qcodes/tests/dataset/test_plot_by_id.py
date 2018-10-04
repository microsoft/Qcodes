import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

import pytest

import qcodes as qc
from qcodes import ParamSpec, new_data_set, new_experiment
from qcodes.dataset.plotting import plot_by_id
from qcodes.dataset.database import initialise_database


@pytest.fixture(scope="function")
def empty_temp_db():
    global n_experiments
    n_experiments = 0
    # create a temp database for testing
    with tempfile.TemporaryDirectory() as tmpdirname:
        qc.config["core"]["db_location"] = os.path.join(tmpdirname, 'temp.db')
        qc.config["core"]["db_debug"] = False
        initialise_database()
        yield


@pytest.fixture(scope='function')
def experiment(empty_temp_db):
    e = new_experiment("test-experiment", sample_name="test-sample")
    yield e
    e.conn.close()


def test_string_stuff(experiment):
    # Let's use the number of the measurement as a 2nd dimension
    meas_number = ParamSpec('meas_number', 'numeric',
                            label='Measurement number', unit='#')

    # The values of the qubit state correspond to 2-qubit correlators
    val = ParamSpec('val', 'numeric', label='Component value', unit='')

    # 2-qubit correlator
    two_q_corr_values = ['X_X', 'X_Y', 'X_Z', 'X_I', 'Y_Y', 'Y_Z', 'Y_I', 'Z_Z',
                         'Z_I']
    #                      note the >>> paramtype <<< it is important
    two_q_corr = ParamSpec('two_q_corr', 'text', label='2-qubit correlator',
                           unit='',
                           depends_on=[val, meas_number])

    data_set = new_data_set('sweep-with-strings')

    data_set.add_parameter(meas_number)
    data_set.add_parameter(val)
    data_set.add_parameter(two_q_corr)

    n_measurements = 20

    for n_meas in range(n_measurements):
        for two_q_corr_ind, two_q_corr_id in enumerate(two_q_corr_values):
            value = two_q_corr_ind + np.random.rand()
            data_set.add_result({'two_q_corr': two_q_corr_id, 'val': value,
                                 'meas_number': n_meas})

    data_set.mark_complete()

    run_id_3d_strings_dep = data_set.run_id

    plt.interactive(False)

    _ = plot_by_id(run_id_3d_strings_dep)

    # plt.show()
    plt.close()
