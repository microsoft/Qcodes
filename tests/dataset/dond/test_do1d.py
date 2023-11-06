import logging

import hypothesis.strategies as hst
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from pytest import LogCaptureFixture

from qcodes import config, validators
from qcodes.dataset import do1d, new_experiment
from qcodes.dataset.data_set import DataSet
from qcodes.instrument_drivers.mock_instruments import (
    ArraySetPointParam,
    Multi2DSetPointParam,
    Multi2DSetPointParam2Sizes,
    MultiSetPointParam,
)
from qcodes.parameters import Parameter
from tests.dataset.conftest import ArrayshapedParam


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("delay", [0, 0.1, 1])
def test_do1d_with_real_parameter(_param_set, _param, delay) -> None:
    start = 0
    stop = 1
    num_points = 1

    do1d(_param_set, start, stop, num_points, delay, _param)


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("plot", [None, True, False])
@pytest.mark.parametrize("plot_config", [None, True, False])
def test_do1d_plot(_param_set, _param, plot, plot_config) -> None:
    if plot_config is not None:
        config.dataset.dond_plot = plot_config

    start = 0
    stop = 1
    num_points = 1

    output = do1d(_param_set, start, stop, num_points, 0, _param, do_plot=plot)
    assert len(output[1]) == 1
    if plot is True or plot is None and plot_config is True:
        assert isinstance(output[1][0], matplotlib.axes.Axes)
    else:
        assert output[1][0] is None


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("delay", [0, 0.1, 1])
def test_do1d_with_complex_parameter(_param_set, _param_complex, delay) -> None:
    start = 0
    stop = 1
    num_points = 1

    do1d(_param_set, start, stop, num_points, delay, _param_complex)


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("delay", [0, 0.1, 1])
def test_do1d_with_2_parameter(_param_set, _param, _param_complex, delay) -> None:
    start = 0
    stop = 1
    num_points = 1

    do1d(_param_set, start, stop, num_points, delay, _param, _param_complex)


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("delay", [0, 0.1, 1])
def test_do1d_output_type_real_parameter(_param_set, _param, delay) -> None:
    start = 0
    stop = 1
    num_points = 1

    data = do1d(_param_set, start, stop, num_points, delay, _param)
    assert isinstance(data[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do1d_output_data(_param, _param_set) -> None:
    start = 0
    stop = 1
    num_points = 5
    delay = 0

    exp = do1d(_param_set, start, stop, num_points, delay, _param)
    data = exp[0]

    assert data.description.interdeps.names == (_param.name, _param_set.name)
    loaded_data = data.get_parameter_data()["simple_parameter"]

    np.testing.assert_array_equal(loaded_data[_param.name], np.ones(5))
    np.testing.assert_array_equal(loaded_data[_param_set.name], np.linspace(0, 1, 5))


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize(
    "multiparamtype",
    [MultiSetPointParam, Multi2DSetPointParam, Multi2DSetPointParam2Sizes],
)
@given(
    num_points=hst.integers(min_value=1, max_value=10),
    n_points_pws=hst.integers(min_value=1, max_value=500),
)
@settings(deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_do1d_verify_shape(
    _param,
    _param_complex,
    _param_set,
    multiparamtype,
    dummyinstrument,
    num_points,
    n_points_pws,
) -> None:
    arrayparam = ArraySetPointParam(name="arrayparam")
    multiparam = multiparamtype(name="multiparam")
    paramwsetpoints = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_start(0)
    dummyinstrument.A.dummy_stop(1)
    dummyinstrument.A.dummy_n_points(n_points_pws)

    start = 0
    stop = 1
    delay = 0

    results = do1d(
        _param_set,
        start,
        stop,
        num_points,
        delay,
        arrayparam,
        multiparam,
        paramwsetpoints,
        _param,
        _param_complex,
        do_plot=False,
    )
    expected_shapes = {}
    for i, name in enumerate(multiparam.full_names):
        expected_shapes[name] = (num_points,) + tuple(multiparam.shapes[i])
    expected_shapes["arrayparam"] = (num_points,) + tuple(arrayparam.shape)
    expected_shapes["simple_parameter"] = (num_points,)
    expected_shapes["simple_complex_parameter"] = (num_points,)
    expected_shapes[paramwsetpoints.full_name] = (num_points, n_points_pws)
    assert results[0].description.shapes == expected_shapes


@pytest.mark.usefixtures("experiment")
def test_do1d_parameter_with_array_vals(_param_set) -> None:
    param = ArrayshapedParam(
        name="paramwitharrayval", vals=validators.Arrays(shape=(10,))
    )
    start = 0
    stop = 1
    num_points = 15  # make param
    delay = 0

    results = do1d(_param_set, start, stop, num_points, delay, param, do_plot=False)
    expected_shapes = {"paramwitharrayval": (num_points, 10)}

    ds = results[0]

    assert ds.description.shapes == expected_shapes

    data = ds.get_parameter_data()

    for name, data_inner in data.items():
        for param_data in data_inner.values():
            assert param_data.shape == expected_shapes[name]


def test_do1d_explicit_experiment(_param_set, _param, experiment) -> None:
    start = 0
    stop = 1
    num_points = 5
    delay = 0

    experiment_2 = new_experiment("new-exp", "no-sample")

    data1 = do1d(
        _param_set,
        start,
        stop,
        num_points,
        delay,
        _param,
        do_plot=False,
        exp=experiment,
    )
    assert data1[0].exp_name == "test-experiment"
    data2 = do1d(
        _param_set,
        start,
        stop,
        num_points,
        delay,
        _param,
        do_plot=False,
        exp=experiment_2,
    )
    assert data2[0].exp_name == "new-exp"
    # by default the last experiment is used
    data3 = do1d(_param_set, start, stop, num_points, delay, _param, do_plot=False)
    assert data3[0].exp_name == "new-exp"


@pytest.mark.usefixtures("experiment")
def test_do1d_explicit_name(_param_set, _param) -> None:
    start = 0
    stop = 1
    num_points = 5
    delay = 0

    data1 = do1d(
        _param_set,
        start,
        stop,
        num_points,
        delay,
        _param,
        do_plot=False,
        measurement_name="my measurement",
    )
    assert data1[0].name == "my measurement"


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do1d_additional_setpoints(_param, _param_complex, _param_set) -> None:
    additional_setpoints = [
        Parameter(f"additional_setter_parameter_{i}", set_cmd=None, get_cmd=None)
        for i in range(2)
    ]
    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 5
    delay_p1 = 0

    for x in range(3):
        for y in range(4):
            additional_setpoints[0](x)
            additional_setpoints[1](y)
            results = do1d(
                _param_set,
                start_p1,
                stop_p1,
                num_points_p1,
                delay_p1,
                _param,
                _param_complex,
                additional_setpoints=additional_setpoints,
            )
            for deps in results[0].description.interdeps.dependencies.values():
                assert len(deps) == 1 + len(additional_setpoints)
            # Calling the fixture won't work here due to loop-scope.
            # Thus, we make an explicit call to close plots. This will be
            # repeated in similarly design tests.
            plt.close("all")


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(num_points_p1=hst.integers(min_value=1, max_value=10))
@pytest.mark.usefixtures("experiment")
def test_do1d_additional_setpoints_shape(
    _param, _param_complex, _param_set, num_points_p1
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

    x = 1
    y = 2

    additional_setpoints[0](x)
    additional_setpoints[1](y)
    results = do1d(
        _param_set,
        start_p1,
        stop_p1,
        num_points_p1,
        delay_p1,
        _param,
        arrayparam,
        additional_setpoints=additional_setpoints,
        do_plot=False,
    )
    expected_shapes = {
        "arrayparam": (num_points_p1, 1, 1, array_shape[0]),
        "simple_parameter": (num_points_p1, 1, 1),
    }
    assert results[0].description.shapes == expected_shapes


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do1d_break_condition(caplog: LogCaptureFixture, _param_set, _param) -> None:
    start = 0
    stop = 1
    num_points = 5
    delay = 0

    def break_condition():
        return True

    data = do1d(
        _param_set,
        start,
        stop,
        num_points,
        delay,
        _param,
        break_condition=break_condition,
    )

    assert isinstance(data[0], DataSet) is True
    assert (
        "qcodes.dataset.dond.do_nd_utils",
        logging.WARNING,
        "Measurement has been interrupted, data may be incomplete: Break condition was met.",
    ) in caplog.record_tuples
