from __future__ import annotations

import numpy as np
import pytest

from qcodes.dataset import do2d_retrace
from qcodes.parameters import Parameter


@pytest.mark.usefixtures("experiment")
def test_do2d_retrace(
    _param_set: Parameter[float | None], _param_set_2: Parameter[float | None]
) -> None:
    def measure() -> float:
        outer = _param_set.cache.get()
        inner = _param_set_2.cache.get()
        assert outer is not None
        assert inner is not None
        return 10 * outer + inner

    meas_param = Parameter("meas_param", set_cmd=None, get_cmd=measure)
    calls: list[str] = []

    trace, retrace = do2d_retrace(
        _param_set,
        0,
        1,
        2,
        0,
        _param_set_2,
        0,
        2,
        3,
        0,
        meas_param,
        enter_actions=(lambda: calls.append("enter"),),
        exit_actions=(lambda: calls.append("exit"),),
        before_inner_actions=(lambda: calls.append("before_inner"),),
        after_inner_actions=(lambda: calls.append("after_inner"),),
        measurement_name="my_measurement",
        show_progress=False,
    )

    assert trace.name == "my_measurement"
    assert retrace.name == "retrace my_measurement"

    outer_setpoints = np.array([0, 0, 0, 1, 1, 1])
    expected_inner = {
        "trace": np.array([0, 1, 2, 0, 1, 2]),
        "retrace": np.array([2, 1, 0, 2, 1, 0]),
    }
    for kind, dataset in (("trace", trace), ("retrace", retrace)):
        data = dataset.get_parameter_data()["meas_param"]
        np.testing.assert_array_equal(data[_param_set.full_name], outer_setpoints)
        np.testing.assert_array_equal(
            data[_param_set_2.full_name], expected_inner[kind]
        )
        np.testing.assert_array_equal(
            data["meas_param"], 10 * outer_setpoints + expected_inner[kind]
        )

    # after_inner_actions are post actions of the inner sweeps so they run
    # after each point of the inner sweeps
    inner_sweep = ["before_inner"] + ["after_inner"] * 3
    assert calls == ["enter"] + inner_sweep * 2 * 2 + ["exit"]


@pytest.mark.usefixtures("experiment")
def test_do2d_retrace_additional_setpoints(
    _param: Parameter[int],
    _param_set: Parameter[float | None],
    _param_set_2: Parameter[float | None],
) -> None:
    additional_setpoint = Parameter(
        "additional_setpoint", set_cmd=None, initial_value=5
    )

    trace, retrace = do2d_retrace(
        _param_set,
        0,
        1,
        2,
        0,
        _param_set_2,
        0,
        1,
        2,
        0,
        _param,
        additional_setpoints=(additional_setpoint,),
        show_progress=False,
    )

    for dataset in (trace, retrace):
        data = dataset.get_parameter_data()[_param.full_name]
        assert set(data) == {
            _param.full_name,
            _param_set.full_name,
            _param_set_2.full_name,
            additional_setpoint.full_name,
        }
        np.testing.assert_array_equal(data[additional_setpoint.full_name], [5] * 4)
        np.testing.assert_array_equal(data[_param.full_name], [1] * 4)
