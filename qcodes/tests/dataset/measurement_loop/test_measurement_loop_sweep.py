import contextlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qcodes import ManualParameter, Parameter
from qcodes.dataset import initialise_or_create_database_at, load_or_create_experiment
from qcodes.dataset.data_set import load_by_id
from qcodes.dataset.measurement_loop import MeasurementLoop, Sweep
from qcodes.utils.dataset.doNd import LinSweep


def test_sweep_1_arg_sequence():
    sequence = [1,2,3]
    sweep = Sweep(sequence, name='sweep_name')
    assert sweep.sequence == sequence

def test_sweep_1_arg_parameter_stop():
    sweep_parameter = ManualParameter('sweep_parameter')

    # Should raise an error since it does not have an initial value
    with pytest.raises(ValueError):
        sweep = Sweep(sweep_parameter, stop=10, num=21)

    sweep_parameter(0)
    sweep = Sweep(sweep_parameter, stop=10, num=21)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))

    sweep_parameter.sweep_defaults = {'num': 21}
    sweep = Sweep(sweep_parameter, stop=10)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))


def test_sweep_1_arg_parameter_around():
    sweep_parameter = ManualParameter('sweep_parameter', initial_value=0)

    sweep = Sweep(sweep_parameter, around=5, num=21)
    assert np.allclose(sweep.sequence, np.linspace(-5, 5, 21))

    sweep_parameter.sweep_defaults = {'num': 21}
    sweep = Sweep(sweep_parameter, around=5)
    assert np.allclose(sweep.sequence, np.linspace(-5, 5, 21))


def test_sweep_2_args_parameter_sequence():
    sweep_parameter = ManualParameter('sweep_parameter', initial_value=0)

    sequence = [1, 2, 3]
    sweep = Sweep(sweep_parameter, sequence)
    assert np.allclose(sweep.sequence, sequence)
    assert sweep.parameter == sweep_parameter


def test_sweep_2_args_parameter_stop():
    sweep_parameter = ManualParameter('sweep_parameter')

    # No initial value
    with pytest.raises(ValueError):
        sweep = Sweep(sweep_parameter, 10)
    with pytest.raises(ValueError):
        sweep = Sweep(sweep_parameter, 10, num=21)

    sweep_parameter(0)
    with pytest.raises(SyntaxError):
        sweep = Sweep(sweep_parameter, 10)

    sweep = Sweep(sweep_parameter, 10, num=21)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))

    sweep_parameter.sweep_defaults = {'num': 21}
    sweep = Sweep(sweep_parameter, 10)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))


def test_sweep_2_args_sequence_name():
    sweep_values = [1, 2, 3]
    with pytest.raises(AssertionError):
        sweep = Sweep(sweep_values)

    sweep = Sweep(sweep_values, "sweep_values")
    assert np.allclose(sweep.sequence, sweep_values)


def test_sweep_3_args_parameter_start_stop():
    sweep_parameter = ManualParameter('sweep_parameter')

    with pytest.raises(SyntaxError):
        sweep = Sweep(sweep_parameter, 0, 10)

    sweep = Sweep(sweep_parameter, 0, 10, num=21)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))

    sweep_values = [1, 2, 3]
    with pytest.raises(AssertionError):
        sweep = Sweep(sweep_values)

    sweep = Sweep(sweep_values, "sweep_values")
    assert np.allclose(sweep.sequence, sweep_values)


def test_sweep_4_args_parameter_start_stop_num():
    sweep_parameter = ManualParameter('sweep_parameter')

    sweep = Sweep(sweep_parameter, 0, 10, 21)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))


def test_sweep_step():
    sweep = Sweep(start=0, stop=10, step=0.5)
    assert np.allclose(sweep.sequence, np.linspace(0, 10, 21))

    # Append final element since it isn't a multiple of 0.5
    sweep = Sweep(start=0, stop=9.9, step=0.5)
    assert np.allclose(sweep.sequence, np.append(np.arange(0, 9.9, 0.5), [9.9]))


def test_error_on_iterate_sweep():
    sweep = Sweep([1,2,3], 'sweep')

    with pytest.raises(RuntimeError):
        iter(sweep)
