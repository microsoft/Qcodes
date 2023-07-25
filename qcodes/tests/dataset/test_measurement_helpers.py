import pytest
import gc
import numpy as np
from pathlib import Path
from functools import partial
from itertools import product
import qcodes as qc
from qcodes.dataset import connect, Measurement, LinSweep
from qcodes.validators import Arrays

from qcodes.parameters import Parameter, ParameterWithSetpoints
from qcodes.dataset.measurement_extensions import (
    complex_measurement_context,
    dond_core,
    LinSweeper,
    DataSetDefinition,
)


@pytest.fixture
def default_params():
    set1 = Parameter("set1", get_cmd=None, set_cmd=None, initial_value=0)
    set2 = Parameter("set2", get_cmd=None, set_cmd=None, initial_value=0)
    set3 = Parameter("set3", get_cmd=None, set_cmd=None, initial_value=0)

    def get_set1():
        return set1()

    def get_sum23():
        return set2() + set3()

    def get_diff23():
        return set2() - set3()

    meas1 = Parameter("meas1", get_cmd=get_set1, set_cmd=False)
    meas2 = Parameter("meas2", get_cmd=get_sum23, set_cmd=False)
    meas3 = Parameter("meas3", get_cmd=get_diff23, set_cmd=False)

    return set1, set2, set3, meas1, meas2, meas3


@pytest.fixture
def pws_params(default_params):
    sweep_start = 0
    sweep_stop = 10
    sweep_points = 11

    def get_pws_results():
        setpoints_arr = np.linspace(sweep_start, sweep_stop, sweep_points)
        return setpoints_arr**2 * set1()

    setpoint_array = Parameter(
        "setpoints",
        get_cmd=partial(np.linspace, sweep_start, sweep_stop, sweep_points),
        vals=Arrays(shape=(sweep_points,)),
    )

    pws1 = ParameterWithSetpoints(
        "pws",
        setpoints=(setpoint_array,),
        get_cmd=get_pws_results,
        vals=Arrays(shape=(sweep_points,)),
    )
    set1, set2, set3, meas1, meas2, meas3 = default_params
    return pws1, set1


@pytest.fixture
def default_database_and_experiment(tmp_path):
    db_path = Path(tmp_path) / "context_tests.db"
    qc.initialise_or_create_database_at(db_path)
    experiment = qc.load_or_create_experiment("context_tests")
    yield experiment
    conn = connect(db_path)
    conn.close()
    gc.collect()


def test_context(default_params, default_database_and_experiment):
    experiment = default_database_and_experiment
    set1, set2, set3, meas1, meas2, meas3 = default_params
    dataset_definition = [
        DataSetDefinition(name="dataset_1", independent=[set1], dependent=[meas1]),
        DataSetDefinition(
            name="dataset_2", independent=[set1, set2, set3], dependent=[meas2, meas3]
        ),
    ]

    with complex_measurement_context(dataset_definition, experiment) as datasavers:
        for val in range(5):
            set1(val)
            meas1_val = meas1()
            datasavers[0].add_result((set1, val), (meas1, meas1_val))
            for val2, val3 in product(range(5), repeat=2):
                set2(val2)
                set3(val3)
                meas2_val = meas2()
                meas3_val = meas3()
                datasavers[1].add_result(
                    (set1, val),
                    (set2, val2),
                    (set3, val3),
                    (meas2, meas2_val),
                    (meas3, meas3_val),
                )
        datasets = [datasaver.dataset for datasaver in datasavers]


def test_dond_core(default_params, default_database_and_experiment):
    experiment = default_database_and_experiment
    set1, set2, set3, meas1, meas2, meas3 = default_params

    core_test_measurement = Measurement(name="core_test_1", exp=experiment)
    core_test_measurement.register_parameter(set1)
    core_test_measurement.register_parameter(meas1, setpoints=[set1])
    with core_test_measurement.run() as datasaver:
        sweep1 = LinSweep(set1, 0, 5, 11, 0.001)
        dond_core(datasaver, sweep1, meas1)

        sweep2 = LinSweep(set1, 10, 20, 100, 0.001)
        dond_core(datasaver, sweep2, meas1)

        dataset = datasaver.dataset


def test_dond_core_and_context(default_params, default_database_and_experiment):
    experiment = default_database_and_experiment
    set1, set2, set3, meas1, meas2, meas3 = default_params

    dataset_definition = [
        DataSetDefinition(
            name="dataset_1", independent=[set1, set2], dependent=[meas1, meas2]
        ),
        DataSetDefinition(
            name="dataset_2", independent=[set1, set3], dependent=[meas1, meas3]
        ),
    ]
    with complex_measurement_context(dataset_definition, experiment) as datasavers:
        for _ in LinSweeper(set1, 0, 10, 11, 0.001):
            sweep1 = LinSweep(set2, 0, 10, 11, 0.001)
            sweep2 = LinSweep(set3, -10, 0, 11, 0.001)
            dond_core(datasavers[0], sweep1, meas1, meas2, additional_setpoints=(set1,))
            dond_core(datasavers[1], sweep2, meas1, meas3, additional_setpoints=(set1,))
        datasets = [datasaver.dataset for datasaver in datasavers]


def test_linsweeper(default_params, default_database_and_experiment):
    experiment = default_database_and_experiment
    set1, set2, set3, meas1, meas2, meas3 = default_params

    dataset_definition = [
        DataSetDefinition(
            name="dataset_1", independent=[set1, set2], dependent=[meas1, meas2]
        )
    ]
    with complex_measurement_context(dataset_definition, experiment) as datasavers:
        for _ in LinSweeper(set1, 0, 10, 11, 0.001):
            sweep1 = LinSweep(set2, 0, 10, 11, 0.001)
            dond_core(datasavers[0], sweep1, meas1, meas2, additional_setpoints=(set1,))
            dond_core(datasavers[0], sweep1, meas1, meas2, additional_setpoints=(set1,))

        datasets = [datasaver.dataset for datasaver in datasavers]


def test_context_with_pws(pws_params, default_database_and_experiment):
    experiment = default_database_and_experiment
    pws1, set1 = pws_params
    dataset_definition = [DataSetDefinition("dataset_1", [set1], [pws1])]
    with complex_measurement_context(dataset_definition, experiment) as datasavers:
        for _ in LinSweeper(set1, 0, 10, 11, 0.001):
            dond_core(datasavers[0], pws1, additional_setpoints=(set1,))

        datasets = [datasaver.dataset for datasaver in datasavers]
