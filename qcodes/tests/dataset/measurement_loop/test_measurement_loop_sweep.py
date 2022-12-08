import numpy as np
import pytest

from qcodes.dataset import LinSweep, MeasurementLoop, Sweep, dond, Iterate
from qcodes.instrument import ManualParameter, Parameter


def test_sweep_1_arg_sequence():
    sequence = [1, 2, 3]
    sweep = Sweep(sequence, name="sweep_name")
    assert sweep.sequence == sequence


def test_sweep_1_arg_parameter_stop():
    sweep_parameter = ManualParameter("sweep_parameter")

    # Should raise an error since it does not have an initial value
    with pytest.raises(ValueError):
        sweep = Sweep(sweep_parameter, stop=10, num=21)

    sweep_parameter(0)
    sweep = Sweep(sweep_parameter, stop=10, num=21)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))

    sweep_parameter.sweep_defaults = {"num": 21}
    sweep = Sweep(sweep_parameter, stop=10)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))


def test_sweep_1_arg_parameter_around():
    sweep_parameter = ManualParameter("sweep_parameter", initial_value=0)

    sweep = Sweep(sweep_parameter, around=5, num=21)
    assert np.allclose(sweep.sequence, np.linspace(-5, 5, 21))

    sweep_parameter.sweep_defaults = {"num": 21}
    sweep = Sweep(sweep_parameter, around=5)
    assert np.allclose(sweep.sequence, np.linspace(-5, 5, 21))


def test_sweep_2_args_parameter_sequence():
    sweep_parameter = ManualParameter("sweep_parameter", initial_value=0)

    sequence = [1, 2, 3]
    sweep = Sweep(sweep_parameter, sequence)
    assert np.allclose(sweep.sequence, sequence)
    assert sweep.parameter == sweep_parameter


def test_sweep_2_args_parameter_stop():
    sweep_parameter = ManualParameter("sweep_parameter")

    # No initial value
    with pytest.raises(ValueError):
        sweep = Sweep(sweep_parameter, stop=10)
    with pytest.raises(ValueError):
        sweep = Sweep(sweep_parameter, stop=10, num=21)

    sweep_parameter(0)
    with pytest.raises(SyntaxError):
        sweep = Sweep(sweep_parameter, 10)

    sweep = Sweep(sweep_parameter, stop=10, num=21)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))

    sweep_parameter.sweep_defaults = {"num": 21}
    sweep = Sweep(sweep_parameter, stop=10)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))


def test_sweep_2_args_sequence_name():
    sweep_values = [1, 2, 3]
    sweep = Sweep(sweep_values)
    assert sweep.name == 'iteration'
    assert sweep.label == 'Iteration'

    sweep = Sweep(sweep_values, "sweep_values")
    assert np.allclose(sweep.sequence, sweep_values)


def test_sweep_3_args_parameter_start_stop():
    sweep_parameter = ManualParameter("sweep_parameter")

    with pytest.raises(SyntaxError):
        sweep = Sweep(sweep_parameter, 0, 10)

    sweep = Sweep(sweep_parameter, 0, 10, num=21)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))

    sweep_values = [1, 2, 3]
    sweep = Sweep(sweep_values)
    assert sweep.name == 'iteration'
    assert sweep.label == 'Iteration'

    sweep = Sweep(sweep_values, "sweep_values")
    assert np.allclose(sweep.sequence, sweep_values)


def test_sweep_4_args_parameter_start_stop_num():
    sweep_parameter = ManualParameter("sweep_parameter")

    sweep = Sweep(sweep_parameter, 0, 10, 21)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))


def test_sweep_step():
    sweep = Sweep(start=0, stop=10, step=0.5)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))

    # Append final element since it isn't a multiple of 0.5
    sweep = Sweep(start=0, stop=9.9, step=0.5)
    assert np.allclose(sweep.sequence, np.append(np.arange(0, 9.9, 0.5), [9.9]))


def test_sweep_len():
    sweep = Sweep(start=0, stop=10, step=0.5)
    assert len(sweep) == 21


def test_error_on_iterate_sweep():
    sweep = Sweep([1, 2, 3], "sweep")

    with pytest.raises(RuntimeError):
        iter(sweep)


@pytest.mark.usefixtures("empty_temp_db", "experiment")
def test_sweep_in_dond():
    set_parameter = ManualParameter("set_param")
    sweep = Sweep(set_parameter, [1, 2, 3])
    get_parameter = Parameter("get_param", get_cmd=set_parameter)

    dataset, _, _ = dond(sweep, get_parameter)
    assert np.allclose(
        dataset.get_parameter_data("get_param")["get_param"]["get_param"], [1, 2, 3]
    )


@pytest.mark.usefixtures("empty_temp_db", "experiment")
def test_sweep_and_linsweep_in_dond():
    set_parameter = ManualParameter("set_param")

    sweep = Sweep(set_parameter, [1, 2, 3])

    set_parameter2 = ManualParameter("set_param2")
    linsweep = LinSweep(set_parameter2, 0, 10, 11)
    get_parameter = Parameter("get_param", get_cmd=set_parameter)

    dataset, _, _ = dond(sweep, linsweep, get_parameter)
    arr = dataset.get_parameter_data("get_param")["get_param"]["get_param"]

    assert np.allclose(arr, np.repeat(np.array([1, 2, 3])[:, np.newaxis], 11, axis=1))


@pytest.mark.usefixtures("empty_temp_db", "experiment")
def test_linsweep_in_MeasurementLoop():
    set_parameter = ManualParameter("set_param")
    get_parameter = ManualParameter("get_param", initial_value=42)

    linsweep = LinSweep(set_parameter, 0, 10, 11)

    sweep = Sweep(linsweep)
    assert sweep.name == "set_param"

    with MeasurementLoop("linsweep_in_MeasurementLoop") as msmt:
        for k, val in enumerate(sweep):
            assert val == k
            msmt.measure(get_parameter)


def test_sweep_execute_sweep_args():
    set_parameter = ManualParameter("set_param")
    sweep = Sweep(set_parameter, [1, 2, 3])
    set_parameter2 = ManualParameter("set_param2")
    other_sweep = Sweep(set_parameter2, [1, 2, 3])

    get_param = Parameter(
        "get_param", get_cmd=lambda: set_parameter() + set_parameter2()
    )

    dataset = sweep.execute(other_sweep, measure_params=get_param)

    arr = dataset.get_parameter_data("get_param")["get_param"]["get_param"]
    assert np.allclose(arr, [[2, 3, 4], [3, 4, 5], [4, 5, 6]])
    print(dataset)


def test_sweep_reverting():
    param = ManualParameter('param', initial_value=42)
    with MeasurementLoop('test_revert') as msmt:
        for val in Sweep(param, range(5), revert=True):
            msmt.measure(val, 'value')

        print(msmt._masked_properties)

        assert param() == 42

        param(41)
    assert param() == 41


def test_iterate():
    param = ManualParameter('param', initial_value=42)

    expected_vals = np.linspace(37, 47, 21)
    for k, val in enumerate(Iterate(param, around=5, num=21)):
        assert val == expected_vals[k]

    assert param() == 42