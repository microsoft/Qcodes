import gc
from functools import partial
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from qcodes.dataset import (
    LinSweep,
    Measurement,
    TogetherSweep,
    connect,
    initialise_or_create_database_at,
    load_or_create_experiment,
)
from qcodes.dataset.measurement_extensions import (
    DataSetDefinition,
    LinSweeper,
    datasaver_builder,
    dond_into,
)
from qcodes.parameters import Parameter, ParameterWithSetpoints
from qcodes.validators import Arrays

if TYPE_CHECKING:
    from collections.abc import Sequence


def assert_dataset_as_expected(
    dataset, dims_dict: dict[str, int], data_vars: "Sequence[str]"
):
    xr_ds = dataset.to_xarray_dataset()
    assert xr_ds.sizes == dims_dict
    assert set(xr_ds.data_vars) == set(data_vars)


@pytest.fixture(name="default_params")
def default_params_():
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


@pytest.fixture(name="pws_params")
def pws_params_(default_params):
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
    set1, _, _, _, _, _ = default_params
    return pws1, set1


@pytest.fixture(name="default_database_and_experiment")
def default_database_and_experiment_(tmp_path):
    db_path = Path(tmp_path) / "context_tests.db"
    initialise_or_create_database_at(db_path)
    experiment = load_or_create_experiment("context_tests")
    yield experiment
    conn = connect(db_path)
    conn.close()
    gc.collect()


def test_context(default_params, default_database_and_experiment):
    _ = default_database_and_experiment
    set1, set2, set3, meas1, meas2, meas3 = default_params
    dataset_definition = [
        DataSetDefinition(name="dataset_1", independent=[set1], dependent=[meas1]),
        DataSetDefinition(
            name="dataset_2", independent=[set1, set2, set3], dependent=[meas2, meas3]
        ),
    ]
    outer_val_range = 5
    inner_val_range = 5
    with datasaver_builder(dataset_definition) as datasavers:
        for val in range(outer_val_range):
            set1(val)
            meas1_val = meas1()
            datasavers[0].add_result((set1, val), (meas1, meas1_val))
            for val2, val3 in product(range(inner_val_range), repeat=2):
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

    assert len(datasets) == 2
    assert_dataset_as_expected(
        datasets[0],
        dims_dict={
            set1.name: outer_val_range,
        },
        data_vars=(meas1.name,),
    )
    assert_dataset_as_expected(
        datasets[1],
        dims_dict={
            set1.name: outer_val_range,
            set2.name: inner_val_range,
            set3.name: inner_val_range,
        },
        data_vars=(meas2.name, meas3.name),
    )


def test_dond_into(default_params, default_database_and_experiment):
    experiment = default_database_and_experiment
    set1, _, _, meas1, _, _ = default_params

    core_test_measurement = Measurement(name="core_test_1", exp=experiment)
    core_test_measurement.register_parameter(set1)
    core_test_measurement.register_parameter(meas1, setpoints=[set1])
    with core_test_measurement.run() as datasaver:
        sweep1 = LinSweep(set1, 0, 5, 11, 0.001)
        dond_into(datasaver, sweep1, meas1)

        sweep2 = LinSweep(set1, 10, 20, 100, 0.001)
        dond_into(datasaver, sweep2, meas1)

        dataset = datasaver.dataset

    assert_dataset_as_expected(
        dataset,
        dims_dict={
            set1.name: 111,
        },
        data_vars=(meas1.name,),
    )


def test_dond_into_and_context(default_params, default_database_and_experiment):
    _ = default_database_and_experiment
    set1, set2, set3, meas1, meas2, meas3 = default_params

    dataset_definition = [
        DataSetDefinition(
            name="dataset_1", independent=[set1, set2], dependent=[meas1, meas2]
        ),
        DataSetDefinition(
            name="dataset_2", independent=[set1, set3], dependent=[meas1, meas3]
        ),
    ]
    with datasaver_builder(dataset_definition) as datasavers:
        for _ in LinSweeper(set1, 0, 10, 11, 0.001):
            sweep1 = LinSweep(set2, 0, 10, 11, 0.001)
            sweep2 = LinSweep(set3, -10, 0, 11, 0.001)
            dond_into(datasavers[0], sweep1, meas1, meas2, additional_setpoints=(set1,))
            dond_into(datasavers[1], sweep2, meas1, meas3, additional_setpoints=(set1,))
        datasets = [datasaver.dataset for datasaver in datasavers]

    assert len(datasets) == 2
    assert_dataset_as_expected(
        datasets[0],
        dims_dict={set1.name: 11, set2.name: 11},
        data_vars=(meas1.name, meas2.name),
    )
    assert_dataset_as_expected(
        datasets[1],
        dims_dict={set1.name: 11, set3.name: 11},
        data_vars=(meas1.name, meas3.name),
    )


def test_linsweeper(default_params, default_database_and_experiment):
    _ = default_database_and_experiment
    set1, set2, _, meas1, meas2, _ = default_params

    dataset_definition = [
        DataSetDefinition(
            name="dataset_1", independent=[set1, set2], dependent=[meas1, meas2]
        )
    ]
    with datasaver_builder(dataset_definition) as datasavers:
        for _ in LinSweeper(set1, 0, 10, 11, 0.001):
            sweep1 = LinSweep(set2, 0, 10, 11, 0.001)
            dond_into(datasavers[0], sweep1, meas1, meas2, additional_setpoints=(set1,))

        datasets = [datasaver.dataset for datasaver in datasavers]

    assert len(datasets) == 1
    assert_dataset_as_expected(
        datasets[0],
        dims_dict={set1.name: 11, set2.name: 11},
        data_vars=(meas1.name, meas2.name),
    )


def test_context_with_pws(pws_params, default_database_and_experiment):
    _ = default_database_and_experiment
    pws1, set1 = pws_params
    dataset_definition = [DataSetDefinition("dataset_1", [set1], [pws1])]
    with datasaver_builder(dataset_definition) as datasavers:
        for _ in LinSweeper(set1, 0, 10, 11, 0.001):
            dond_into(datasavers[0], pws1, additional_setpoints=(set1,))

        datasets = [datasaver.dataset for datasaver in datasavers]

    assert len(datasets) == 1
    assert_dataset_as_expected(
        datasets[0],
        dims_dict={
            set1.name: 11,
            pws1.setpoints[0].name: pws1.setpoints[0].vals.shape[0],
        },
        data_vars=(pws1.name,),
    )


def test_dond_into_with_callables(
    default_params, default_database_and_experiment, mocker
):
    experiment = default_database_and_experiment
    set1, _, _, meas1, _, _ = default_params

    core_test_measurement = Measurement(name="core_test_1", exp=experiment)
    core_test_measurement.register_parameter(set1)
    core_test_measurement.register_parameter(meas1, setpoints=[set1])

    internal_callable = mocker.MagicMock(return_value=None)

    with core_test_measurement.run() as datasaver:
        sweep1 = LinSweep(set1, 0, 5, 11, 0.001)
        dond_into(datasaver, sweep1, internal_callable, meas1)

        sweep2 = LinSweep(set1, 10, 20, 100, 0.001)
        dond_into(datasaver, sweep2, internal_callable, meas1)

        dataset = datasaver.dataset

    assert internal_callable.call_count == 11 + 100
    assert_dataset_as_expected(
        dataset,
        dims_dict={
            set1.name: 111,
        },
        data_vars=(meas1.name,),
    )


def test_dond_into_fails_with_together_sweeps(
    default_params, default_database_and_experiment
):
    experiment = default_database_and_experiment
    set1, set2, _, meas1, _, _ = default_params

    core_test_measurement = Measurement(name="core_test_1", exp=experiment)
    core_test_measurement.register_parameter(set1)
    core_test_measurement.register_parameter(meas1, setpoints=[set1])
    with pytest.raises(ValueError, match="dond_into does not support TogetherSweeps"):
        with core_test_measurement.run() as datasaver:
            sweep1 = LinSweep(set1, 0, 5, 11, 0.001)
            sweep2 = LinSweep(set2, 10, 20, 11, 0.001)

            dond_into(
                datasaver,
                TogetherSweep(sweep1, sweep2),  # pyright: ignore [reportArgumentType]
                meas1,
            )
            _ = datasaver.dataset


def test_dond_into_fails_with_groups(default_params, default_database_and_experiment):
    experiment = default_database_and_experiment
    set1, _, _, meas1, meas2, _ = default_params

    core_test_measurement = Measurement(name="core_test_1", exp=experiment)
    core_test_measurement.register_parameter(set1)
    core_test_measurement.register_parameter(meas1, setpoints=[set1])
    with pytest.raises(
        ValueError, match="dond_into does not support multiple datasets"
    ):
        with core_test_measurement.run() as datasaver:
            sweep1 = LinSweep(set1, 0, 5, 11, 0.001)
            dond_into(
                datasaver,
                sweep1,
                [meas1],  # pyright: ignore [reportArgumentType]
                [meas2],  # pyright: ignore [reportArgumentType]
            )
            _ = datasaver.dataset


def test_context_with_multiple_experiments(
    default_params, default_database_and_experiment
):
    experiment = default_database_and_experiment
    experiment2 = load_or_create_experiment("other_experiment")
    set1, set2, set3, meas1, meas2, meas3 = default_params

    dataset_definition = [
        DataSetDefinition(
            name="dataset_1",
            independent=[set1, set2],
            dependent=[meas1, meas2],
            experiment=experiment,
        ),
        DataSetDefinition(
            name="dataset_2",
            independent=[set1, set3],
            dependent=[meas1, meas3],
            experiment=experiment2,
        ),
    ]
    with datasaver_builder(dataset_definition) as datasavers:
        for _ in LinSweeper(set1, 0, 10, 11, 0.001):
            sweep1 = LinSweep(set2, 0, 10, 11, 0.001)
            sweep2 = LinSweep(set3, -10, 0, 11, 0.001)
            dond_into(datasavers[0], sweep1, meas1, meas2, additional_setpoints=(set1,))
            dond_into(datasavers[1], sweep2, meas1, meas3, additional_setpoints=(set1,))
        datasets = [datasaver.dataset for datasaver in datasavers]

    assert len(datasets) == 2
    assert_dataset_as_expected(
        datasets[0],
        dims_dict={set1.name: 11, set2.name: 11},
        data_vars=(meas1.name, meas2.name),
    )
    assert_dataset_as_expected(
        datasets[1],
        dims_dict={set1.name: 11, set3.name: 11},
        data_vars=(meas1.name, meas3.name),
    )

    assert datasets[0].exp_id != datasets[1].exp_id
    assert datasets[0].exp_name == experiment.name
    assert datasets[1].exp_name == experiment2.name


def test_context_with_no_experiment(default_params, default_database_and_experiment):
    _ = default_database_and_experiment
    experiment2 = load_or_create_experiment("other_experiment")
    set1, set2, set3, meas1, meas2, meas3 = default_params

    dataset_definition = [
        DataSetDefinition(
            name="dataset_1",
            independent=[set1, set2],
            dependent=[meas1, meas2],
        ),
        DataSetDefinition(
            name="dataset_2",
            independent=[set1, set3],
            dependent=[meas1, meas3],
        ),
    ]
    with datasaver_builder(dataset_definition) as datasavers:
        for _ in LinSweeper(set1, 0, 10, 11, 0.001):
            sweep1 = LinSweep(set2, 0, 10, 11, 0.001)
            sweep2 = LinSweep(set3, -10, 0, 11, 0.001)
            dond_into(datasavers[0], sweep1, meas1, meas2, additional_setpoints=(set1,))
            dond_into(datasavers[1], sweep2, meas1, meas3, additional_setpoints=(set1,))
        datasets = [datasaver.dataset for datasaver in datasavers]

    assert len(datasets) == 2
    assert_dataset_as_expected(
        datasets[0],
        dims_dict={set1.name: 11, set2.name: 11},
        data_vars=(meas1.name, meas2.name),
    )
    assert_dataset_as_expected(
        datasets[1],
        dims_dict={set1.name: 11, set3.name: 11},
        data_vars=(meas1.name, meas3.name),
    )

    assert datasets[0].exp_name == experiment2.name
    assert datasets[1].exp_name == experiment2.name


def test_context_with_override_experiment(
    default_params, default_database_and_experiment
):
    experiment = default_database_and_experiment
    experiment2 = load_or_create_experiment("other_experiment")
    set1, set2, set3, meas1, meas2, meas3 = default_params

    dataset_definition = [
        DataSetDefinition(
            name="dataset_1",
            independent=[set1, set2],
            dependent=[meas1, meas2],
            experiment=experiment,
        ),
        DataSetDefinition(
            name="dataset_2",
            independent=[set1, set3],
            dependent=[meas1, meas3],
            experiment=experiment,
        ),
    ]
    with datasaver_builder(
        dataset_definition, override_experiment=experiment2
    ) as datasavers:
        for _ in LinSweeper(set1, 0, 10, 11, 0.001):
            sweep1 = LinSweep(set2, 0, 10, 11, 0.001)
            sweep2 = LinSweep(set3, -10, 0, 11, 0.001)
            dond_into(datasavers[0], sweep1, meas1, meas2, additional_setpoints=(set1,))
            dond_into(datasavers[1], sweep2, meas1, meas3, additional_setpoints=(set1,))
        datasets = [datasaver.dataset for datasaver in datasavers]

    assert len(datasets) == 2
    assert_dataset_as_expected(
        datasets[0],
        dims_dict={set1.name: 11, set2.name: 11},
        data_vars=(meas1.name, meas2.name),
    )
    assert_dataset_as_expected(
        datasets[1],
        dims_dict={set1.name: 11, set3.name: 11},
        data_vars=(meas1.name, meas3.name),
    )

    assert datasets[0].exp_name == experiment2.name
    assert datasets[1].exp_name == experiment2.name
