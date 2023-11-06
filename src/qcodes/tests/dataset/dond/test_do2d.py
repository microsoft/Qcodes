"""
These are the basic black box tests for the doNd functions.
"""
import hypothesis.strategies as hst
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from qcodes import config
from qcodes.dataset import do2d, new_experiment
from qcodes.dataset.data_set import DataSet
from qcodes.instrument_drivers.mock_instruments import (
    ArraySetPointParam,
    Multi2DSetPointParam,
    Multi2DSetPointParam2Sizes,
    MultiSetPointParam,
)
from qcodes.parameters import Parameter


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize(
    "sweep, columns", [(False, False), (False, True), (True, False), (True, True)]
)
def test_do2d(_param, _param_complex, _param_set, _param_set_2, sweep, columns) -> None:

    start_p1 = 0
    stop_p1 = 1
    num_points_p1 = 1
    delay_p1 = 0

    start_p2 = 0.1
    stop_p2 = 1.1
    num_points_p2 = 2
    delay_p2 = 0.01

    do2d(
        _param_set,
        start_p1,
        stop_p1,
        num_points_p1,
        delay_p1,
        _param_set_2,
        start_p2,
        stop_p2,
        num_points_p2,
        delay_p2,
        _param,
        _param_complex,
        set_before_sweep=sweep,
        flush_columns=columns,
    )


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("plot", [None, True, False])
@pytest.mark.parametrize("plot_config", [None, True, False])
def test_do2d_plot(_param_set, _param_set_2, _param, plot, plot_config) -> None:

    if plot_config is not None:
        config.dataset.dond_plot = plot_config

    start_p1 = 0
    stop_p1 = 1
    num_points_p1 = 1
    delay_p1 = 0

    start_p2 = 0.1
    stop_p2 = 1.1
    num_points_p2 = 2
    delay_p2 = 0

    output = do2d(
        _param_set,
        start_p1,
        stop_p1,
        num_points_p1,
        delay_p1,
        _param_set_2,
        start_p2,
        stop_p2,
        num_points_p2,
        delay_p2,
        _param,
        do_plot=plot,
    )

    assert len(output[1]) == 1
    if plot is True or plot is None and plot_config is True:
        assert isinstance(output[1][0], matplotlib.axes.Axes)
    else:
        assert output[1][0] is None


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do2d_output_type(_param, _param_complex, _param_set, _param_set_2) -> None:

    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 1
    delay_p1 = 0

    start_p2 = 0.1
    stop_p2 = 0.75
    num_points_p2 = 2
    delay_p2 = 0.025

    data = do2d(
        _param_set,
        start_p1,
        stop_p1,
        num_points_p1,
        delay_p1,
        _param_set_2,
        start_p2,
        stop_p2,
        num_points_p2,
        delay_p2,
        _param,
        _param_complex,
    )
    assert isinstance(data[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do2d_output_data(_param, _param_complex, _param_set, _param_set_2) -> None:

    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 5
    delay_p1 = 0

    start_p2 = 0.5
    stop_p2 = 1
    num_points_p2 = 5
    delay_p2 = 0.0

    exp = do2d(
        _param_set,
        start_p1,
        stop_p1,
        num_points_p1,
        delay_p1,
        _param_set_2,
        start_p2,
        stop_p2,
        num_points_p2,
        delay_p2,
        _param,
        _param_complex,
    )
    data = exp[0]

    assert set(data.description.interdeps.names) == {
        _param.name,
        _param_complex.name,
        _param_set.name,
        _param_set_2.name,
    }
    loaded_data = data.get_parameter_data()
    expected_data_1 = np.ones(25).reshape(num_points_p1, num_points_p2)

    np.testing.assert_array_equal(
        loaded_data[_param.name][_param.name], expected_data_1
    )
    expected_data_2 = (1 + 1j) * np.ones(25).reshape(num_points_p1, num_points_p2)
    np.testing.assert_array_equal(
        loaded_data[_param_complex.name][_param_complex.name], expected_data_2
    )

    expected_setpoints_1 = np.repeat(
        np.linspace(start_p1, stop_p1, num_points_p1), num_points_p2
    ).reshape(num_points_p1, num_points_p2)
    np.testing.assert_array_equal(
        loaded_data[_param_complex.name][_param_set.name], expected_setpoints_1
    )

    expected_setpoints_2 = np.tile(
        np.linspace(start_p2, stop_p2, num_points_p2), num_points_p1
    ).reshape(num_points_p1, num_points_p2)
    np.testing.assert_array_equal(
        loaded_data[_param_complex.name][_param_set_2.name], expected_setpoints_2
    )


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize(
    "sweep, columns", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize(
    "multiparamtype",
    [MultiSetPointParam, Multi2DSetPointParam, Multi2DSetPointParam2Sizes],
)
@given(
    num_points_p1=hst.integers(min_value=1, max_value=5),
    num_points_p2=hst.integers(min_value=1, max_value=5),
    n_points_pws=hst.integers(min_value=1, max_value=500),
)
@settings(deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_do2d_verify_shape(
    _param,
    _param_complex,
    _param_set,
    _param_set_2,
    multiparamtype,
    dummyinstrument,
    sweep,
    columns,
    num_points_p1,
    num_points_p2,
    n_points_pws,
) -> None:
    arrayparam = ArraySetPointParam(name="arrayparam")
    multiparam = multiparamtype(name="multiparam")
    paramwsetpoints = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_start(0)
    dummyinstrument.A.dummy_stop(1)
    dummyinstrument.A.dummy_n_points(n_points_pws)

    start_p1 = 0
    stop_p1 = 1
    delay_p1 = 0

    start_p2 = 0.1
    stop_p2 = 1.1
    delay_p2 = 0

    results = do2d(
        _param_set,
        start_p1,
        stop_p1,
        num_points_p1,
        delay_p1,
        _param_set_2,
        start_p2,
        stop_p2,
        num_points_p2,
        delay_p2,
        arrayparam,
        multiparam,
        paramwsetpoints,
        _param,
        _param_complex,
        set_before_sweep=sweep,
        flush_columns=columns,
        do_plot=False,
    )
    expected_shapes = {}
    for i, name in enumerate(multiparam.full_names):
        expected_shapes[name] = (num_points_p1, num_points_p2) + tuple(
            multiparam.shapes[i]
        )
    expected_shapes["arrayparam"] = (num_points_p1, num_points_p2) + tuple(
        arrayparam.shape
    )
    expected_shapes["simple_parameter"] = (num_points_p1, num_points_p2)
    expected_shapes["simple_complex_parameter"] = (num_points_p1, num_points_p2)
    expected_shapes[paramwsetpoints.full_name] = (
        num_points_p1,
        num_points_p2,
        n_points_pws,
    )

    assert results[0].description.shapes == expected_shapes
    ds = results[0]

    data = ds.get_parameter_data()

    for name, data_inner in data.items():
        for param_data in data_inner.values():
            assert param_data.shape == expected_shapes[name]


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do2d_additional_setpoints(
    _param, _param_complex, _param_set, _param_set_2
) -> None:
    additional_setpoints = [
        Parameter(f"additional_setter_parameter_{i}", set_cmd=None, get_cmd=None)
        for i in range(2)
    ]
    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 5
    delay_p1 = 0

    start_p2 = 0.5
    stop_p2 = 1
    num_points_p2 = 5
    delay_p2 = 0.0
    for x in range(3):
        for y in range(4):
            additional_setpoints[0](x)
            additional_setpoints[1](y)
            results = do2d(
                _param_set,
                start_p1,
                stop_p1,
                num_points_p1,
                delay_p1,
                _param_set_2,
                start_p2,
                stop_p2,
                num_points_p2,
                delay_p2,
                _param,
                _param_complex,
                additional_setpoints=additional_setpoints,
            )
            for deps in results[0].description.interdeps.dependencies.values():
                assert len(deps) == 2 + len(additional_setpoints)
            plt.close("all")


@given(
    num_points_p1=hst.integers(min_value=1, max_value=5),
    num_points_p2=hst.integers(min_value=1, max_value=5),
)
@settings(deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
@pytest.mark.usefixtures("experiment")
def test_do2d_additional_setpoints_shape(
    _param, _param_complex, _param_set, _param_set_2, num_points_p1, num_points_p2
) -> None:
    arrayparam = ArraySetPointParam(name="arrayparam")
    array_shape = arrayparam.shape
    additional_setpoints = [
        Parameter(f"additional_setter_parameter_{i}", set_cmd=None, get_cmd=None)
        for i in range(2)
    ]
    start_p1 = 0
    stop_p1 = 0.5
    delay_p1 = 0

    start_p2 = 0.5
    stop_p2 = 1
    delay_p2 = 0.0

    x = 1
    y = 2

    additional_setpoints[0](x)
    additional_setpoints[1](y)
    results = do2d(
        _param_set,
        start_p1,
        stop_p1,
        num_points_p1,
        delay_p1,
        _param_set_2,
        start_p2,
        stop_p2,
        num_points_p2,
        delay_p2,
        _param,
        _param_complex,
        arrayparam,
        additional_setpoints=additional_setpoints,
        do_plot=False,
    )
    expected_shapes = {
        "arrayparam": (num_points_p1, num_points_p2, 1, 1, array_shape[0]),
        "simple_complex_parameter": (num_points_p1, num_points_p2, 1, 1),
        "simple_parameter": (num_points_p1, num_points_p2, 1, 1),
    }
    assert results[0].description.shapes == expected_shapes


def test_do2d_explicit_experiment(_param_set, _param_set_2, _param, experiment) -> None:
    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 5
    delay_p1 = 0

    start_p2 = 0.5
    stop_p2 = 1
    num_points_p2 = 5
    delay_p2 = 0.0

    experiment_2 = new_experiment("new-exp", "no-sample")

    data1 = do2d(
        _param_set,
        start_p1,
        stop_p1,
        num_points_p1,
        delay_p1,
        _param_set_2,
        start_p2,
        stop_p2,
        num_points_p2,
        delay_p2,
        _param,
        do_plot=False,
        exp=experiment,
    )
    assert data1[0].exp_name == "test-experiment"
    data2 = do2d(
        _param_set,
        start_p1,
        stop_p1,
        num_points_p1,
        delay_p1,
        _param_set_2,
        start_p2,
        stop_p2,
        num_points_p2,
        delay_p2,
        _param,
        do_plot=False,
        exp=experiment_2,
    )
    assert data2[0].exp_name == "new-exp"
    # by default the last experiment is used
    data3 = do2d(
        _param_set,
        start_p1,
        stop_p1,
        num_points_p1,
        delay_p1,
        _param_set_2,
        start_p2,
        stop_p2,
        num_points_p2,
        delay_p2,
        _param,
        do_plot=False,
    )
    assert data3[0].exp_name == "new-exp"


@pytest.mark.usefixtures("experiment")
def test_do2d_explicit_name(_param_set, _param_set_2, _param) -> None:
    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 5
    delay_p1 = 0

    start_p2 = 0.5
    stop_p2 = 1
    num_points_p2 = 5
    delay_p2 = 0.0

    data1 = do2d(
        _param_set,
        start_p1,
        stop_p1,
        num_points_p1,
        delay_p1,
        _param_set_2,
        start_p2,
        stop_p2,
        num_points_p2,
        delay_p2,
        _param,
        do_plot=False,
        measurement_name="my measurement",
    )
    assert data1[0].name == "my measurement"
