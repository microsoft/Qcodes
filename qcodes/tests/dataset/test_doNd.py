"""
These are the basic black box tests for the doNd functions.
"""
import hypothesis.strategies as hst
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from qcodes import config
from qcodes.dataset import new_experiment
from qcodes.dataset.data_set import DataSet
from qcodes.instrument.parameter import Parameter, _BaseParameter
from qcodes.tests.instrument_mocks import (
    ArraySetPointParam,
    Multi2DSetPointParam,
    Multi2DSetPointParam2Sizes,
    MultiSetPointParam,
)
from qcodes.utils import validators
from qcodes.utils.dataset.doNd import ArraySweep, LinSweep, LogSweep, do0d, do1d, do2d, dond
from qcodes.utils.validators import Arrays

from .conftest import ArrayshapedParam


@pytest.fixture(autouse=True)
def set_tmp_output_dir(tmpdir):
    old_config = config.user.mainfolder
    try:
        config.user.mainfolder = str(tmpdir)
        yield
    finally:
        config.user.mainfolder = old_config


@pytest.fixture()
def plot_close():
    yield
    plt.close("all")


@pytest.fixture()
def _param():
    p = Parameter("simple_parameter", set_cmd=None, get_cmd=lambda: 1)
    return p


@pytest.fixture()
def _param_2():
    p = Parameter("simple_parameter", set_cmd=None, get_cmd=lambda: 2)
    return p


@pytest.fixture()
def _param_complex():
    p = Parameter(
        "simple_complex_parameter",
        set_cmd=None,
        get_cmd=lambda: 1 + 1j,
        vals=validators.ComplexNumbers(),
    )
    return p


@pytest.fixture()
def _param_complex_2():
    p = Parameter(
        "simple_complex_parameter",
        set_cmd=None,
        get_cmd=lambda: 2 + 2j,
        vals=validators.ComplexNumbers(),
    )
    return p


@pytest.fixture()
def _param_set():
    p = Parameter("simple_setter_parameter", set_cmd=None, get_cmd=None)
    return p


@pytest.fixture()
def _param_set_2():
    p = Parameter("simple_setter_parameter_2", set_cmd=None, get_cmd=None)
    return p


def _param_func(_p):
    """
    A private utility function.
    """
    _new_param = Parameter(
        "modified_parameter", set_cmd=None, get_cmd=lambda: _p.get() * 2
    )
    return _new_param


@pytest.fixture()
def _param_callable(_param):
    return _param_func(_param)


def test_param_callable(_param_callable):
    _param_modified = _param_callable
    assert _param_modified.get() == 2


@pytest.fixture()
def _string_callable():
    return "Call"


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("period", [None, 1])
@pytest.mark.parametrize("plot", [None, True, False])
@pytest.mark.parametrize("plot_config", [None, True, False])
def test_do0d_with_real_parameter(period, plot, plot_config):
    arrayparam = ArraySetPointParam(name="arrayparam")

    if plot_config is not None:
        config.dataset.dond_plot = plot_config

    output = do0d(arrayparam, write_period=period, do_plot=plot)
    assert len(output[1]) == 1
    if plot is True or plot is None and plot_config is True:
        assert isinstance(output[1][0], matplotlib.axes.Axes)
    else:
        assert output[1][0] is None


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize(
    "period, plot", [(None, True), (None, False), (1, True), (1, False)]
)
def test_do0d_with_complex_parameter(_param_complex, period, plot):
    do0d(_param_complex, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize(
    "period, plot", [(None, True), (None, False), (1, True), (1, False)]
)
def test_do0d_with_a_callable(_param_callable, period, plot):
    do0d(_param_callable, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize(
    "period, plot", [(None, True), (None, False), (1, True), (1, False)]
)
def test_do0d_with_2_parameters(_param, _param_complex, period, plot):
    do0d(_param, _param_complex, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize(
    "period, plot", [(None, True), (None, False), (1, True), (1, False)]
)
def test_do0d_with_parameter_and_a_callable(
    _param_complex, _param_callable, period, plot
):
    do0d(_param_callable, _param_complex, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do0d_output_type_real_parameter(_param):
    data = do0d(_param)
    assert isinstance(data[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do0d_output_type_complex_parameter(_param_complex):
    data_complex = do0d(_param_complex)
    assert isinstance(data_complex[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do0d_output_type_callable(_param_callable):
    data_func = do0d(_param_callable)
    assert isinstance(data_func[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do0d_output_data(_param):
    exp = do0d(_param)
    data = exp[0]
    assert data.parameters == _param.name
    loaded_data = data.get_parameter_data()["simple_parameter"]["simple_parameter"]
    assert loaded_data == np.array([_param.get()])


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize(
    "multiparamtype",
    [MultiSetPointParam, Multi2DSetPointParam, Multi2DSetPointParam2Sizes],
)
@given(n_points_pws=hst.integers(min_value=1, max_value=1000))
@settings(deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_do0d_verify_shape(
    _param, _param_complex, multiparamtype, dummyinstrument, n_points_pws
):
    arrayparam = ArraySetPointParam(name="arrayparam")
    multiparam = multiparamtype(name="multiparam")
    paramwsetpoints = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_start(0)
    dummyinstrument.A.dummy_stop(1)
    dummyinstrument.A.dummy_n_points(n_points_pws)

    results = do0d(
        arrayparam, multiparam, paramwsetpoints, _param, _param_complex, do_plot=False
    )
    expected_shapes = {}
    for i, name in enumerate(multiparam.full_names):
        expected_shapes[name] = tuple(multiparam.shapes[i])
    expected_shapes["arrayparam"] = tuple(arrayparam.shape)
    expected_shapes["simple_parameter"] = (1,)
    expected_shapes["simple_complex_parameter"] = (1,)
    expected_shapes[paramwsetpoints.full_name] = (n_points_pws,)
    assert results[0].description.shapes == expected_shapes

    ds = results[0]

    assert ds.description.shapes == expected_shapes

    data = ds.get_parameter_data()

    for name, data in data.items():
        for param_data in data.values():
            assert param_data.shape == expected_shapes[name]


@pytest.mark.usefixtures("experiment")
def test_do0d_parameter_with_array_vals():
    param = ArrayshapedParam(name="paramwitharrayval", vals=Arrays(shape=(10,)))
    results = do0d(param)
    expected_shapes = {"paramwitharrayval": (10,)}
    assert results[0].description.shapes == expected_shapes


def test_do0d_explicit_experiment(_param, experiment):
    experiment_2 = new_experiment("new-exp", "no-sample")

    data1 = do0d(_param, do_plot=False, exp=experiment)
    assert data1[0].exp_name == "test-experiment"
    data2 = do0d(_param, do_plot=False, exp=experiment_2)
    assert data2[0].exp_name == "new-exp"
    # by default the last experiment is used
    data3 = do0d(_param, do_plot=False)
    assert data3[0].exp_name == "new-exp"


@pytest.mark.usefixtures("experiment")
def test_do0d_explicit_name(_param):
    data1 = do0d(_param, do_plot=False, measurement_name="my measurement")
    assert data1[0].name == "my measurement"


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("delay", [0, 0.1, 1])
def test_do1d_with_real_parameter(_param_set, _param, delay):

    start = 0
    stop = 1
    num_points = 1

    do1d(_param_set, start, stop, num_points, delay, _param)


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("plot", [None, True, False])
@pytest.mark.parametrize("plot_config", [None, True, False])
def test_do1d_plot(_param_set, _param, plot, plot_config):

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
def test_do1d_with_complex_parameter(_param_set, _param_complex, delay):

    start = 0
    stop = 1
    num_points = 1

    do1d(_param_set, start, stop, num_points, delay, _param_complex)


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("delay", [0, 0.1, 1])
def test_do1d_with_2_parameter(_param_set, _param, _param_complex, delay):

    start = 0
    stop = 1
    num_points = 1

    do1d(_param_set, start, stop, num_points, delay, _param, _param_complex)


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("delay", [0, 0.1, 1])
def test_do1d_output_type_real_parameter(_param_set, _param, delay):

    start = 0
    stop = 1
    num_points = 1

    data = do1d(_param_set, start, stop, num_points, delay, _param)
    assert isinstance(data[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do1d_output_data(_param, _param_set):

    start = 0
    stop = 1
    num_points = 5
    delay = 0

    exp = do1d(_param_set, start, stop, num_points, delay, _param)
    data = exp[0]

    assert data.parameters == f"{_param_set.name},{_param.name}"
    loaded_data = data.get_parameter_data()["simple_parameter"]

    np.testing.assert_array_equal(loaded_data[_param.name], np.ones(5))
    np.testing.assert_array_equal(loaded_data[_param_set.name], np.linspace(0, 1, 5))


def test_do0d_parameter_with_setpoints_2d(dummyinstrument):
    dummyinstrument.A.dummy_start(0)
    dummyinstrument.A.dummy_stop(10)
    dummyinstrument.A.dummy_n_points(10)
    dummyinstrument.A.dummy_start_2(2)
    dummyinstrument.A.dummy_stop_2(7)
    dummyinstrument.A.dummy_n_points_2(3)
    dataset, _, _ = do0d(dummyinstrument.A.dummy_parameter_with_setpoints_2d)

    data = dataset.cache.data()[
        "dummyinstrument_ChanA_dummy_parameter_with_setpoints_2d"
    ]
    for array in data.values():
        assert array.shape == (10, 3)


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
):
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
def test_do1d_parameter_with_array_vals(_param_set):
    param = ArrayshapedParam(name="paramwitharrayval", vals=Arrays(shape=(10,)))
    start = 0
    stop = 1
    num_points = 15  # make param
    delay = 0

    results = do1d(_param_set, start, stop, num_points, delay, param, do_plot=False)
    expected_shapes = {"paramwitharrayval": (num_points, 10)}

    ds = results[0]

    assert ds.description.shapes == expected_shapes

    data = ds.get_parameter_data()

    for name, data in data.items():
        for param_data in data.values():
            assert param_data.shape == expected_shapes[name]


def test_do1d_explicit_experiment(_param_set, _param, experiment):
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
def test_do1d_explicit_name(_param_set, _param):
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
@pytest.mark.parametrize(
    "sweep, columns", [(False, False), (False, True), (True, False), (True, True)]
)
def test_do2d(_param, _param_complex, _param_set, _param_set_2, sweep, columns):

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
def test_do2d_plot(_param_set, _param_set_2, _param, plot, plot_config):

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
def test_do2d_output_type(_param, _param_complex, _param_set, _param_set_2):

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
def test_do2d_output_data(_param, _param_complex, _param_set, _param_set_2):

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

    assert data.parameters == (
        f"{_param_set.name},{_param_set_2.name}," f"{_param.name},{_param_complex.name}"
    )
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
):
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

    for name, data in data.items():
        for param_data in data.values():
            assert param_data.shape == expected_shapes[name]


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do1d_additional_setpoints(_param, _param_complex, _param_set):
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
):
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
def test_do2d_additional_setpoints(_param, _param_complex, _param_set, _param_set_2):
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
):
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


def test_do2d_explicit_experiment(_param_set, _param_set_2, _param, experiment):
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
def test_do2d_explicit_name(_param_set, _param_set_2, _param):
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


def test_linear_sweep_get_setpoints(_param):
    start = 0
    stop = 1
    num_points = 5
    delay = 1
    sweep = LinSweep(_param, start, stop, num_points, delay)

    np.testing.assert_array_equal(
        sweep.get_setpoints(), np.linspace(start, stop, num_points)
    )


def test_linear_sweep_properties(_param, _param_complex):
    start = 0
    stop = 1
    num_points = 5
    delay = 1
    sweep = LinSweep(_param, start, stop, num_points, delay)
    assert isinstance(sweep.param, _BaseParameter)
    assert sweep.delay == delay
    assert sweep.param == _param
    assert sweep.num_points == num_points

    # test default delay 0 with complex param
    sweep_2 = LinSweep(_param_complex, start, stop, num_points)
    assert sweep_2.delay == 0


def test_linear_sweep_parameter_class(_param, _param_complex):
    start = 0
    stop = 1
    num_points = 5
    delay = 1
    sweep = LinSweep(_param, start, stop, num_points, delay)
    assert isinstance(sweep.param, _BaseParameter)

    sweep_2 = LinSweep(_param_complex, start, stop, num_points)
    assert isinstance(sweep_2.param, _BaseParameter)

    arrayparam = ArraySetPointParam(name="arrayparam")
    sweep_3 = LinSweep(arrayparam, start, stop, num_points)
    assert isinstance(sweep_3.param, _BaseParameter)


def test_log_sweep_get_setpoints(_param):
    start = 0
    stop = 1
    num_points = 5
    delay = 1
    sweep = LogSweep(_param, start, stop, num_points, delay)

    np.testing.assert_array_equal(
        sweep.get_setpoints(), np.logspace(start, stop, num_points)
    )


def test_log_sweep_properties(_param, _param_complex):
    start = 0
    stop = 1
    num_points = 5
    delay = 1
    sweep = LogSweep(_param, start, stop, num_points, delay)
    assert isinstance(sweep.param, _BaseParameter)
    assert sweep.delay == delay
    assert sweep.param == _param
    assert sweep.num_points == num_points

    # test default delay 0 with complex param
    sweep_2 = LogSweep(_param_complex, start, stop, num_points)
    assert sweep_2.delay == 0


def test_log_sweep_parameter_class(_param, _param_complex):
    start = 0
    stop = 1
    num_points = 5
    delay = 1
    sweep = LogSweep(_param, start, stop, num_points, delay)
    assert isinstance(sweep.param, _BaseParameter)

    sweep_2 = LogSweep(_param_complex, start, stop, num_points)
    assert isinstance(sweep_2.param, _BaseParameter)

    arrayparam = ArraySetPointParam(name="arrayparam")
    sweep_3 = LogSweep(arrayparam, start, stop, num_points)
    assert isinstance(sweep_3.param, _BaseParameter)


def test_array_sweep_get_setpoints(_param):
    array = np.linspace(0, 1, 5)
    delay = 1
    sweep = ArraySweep(_param, array, delay)

    np.testing.assert_array_equal(
        sweep.get_setpoints(), array
    )

    array2 = [1, 2, 3, 4, 5, 5.2]
    sweep2 = ArraySweep(_param, array2)

    np.testing.assert_array_equal(
        sweep2.get_setpoints(), np.array(array2)
    )


def test_array_sweep_properties(_param):
    array = np.linspace(0, 1, 5)
    delay = 1
    sweep = ArraySweep(_param, array, delay)
    assert isinstance(sweep.param, _BaseParameter)
    assert sweep.delay == delay
    assert sweep.param == _param
    assert sweep.num_points == len(array)

    # test default delay 0
    sweep_2 = ArraySweep(_param, array)
    assert sweep_2.delay == 0


def test_array_sweep_parameter_class(_param, _param_complex):
    array = np.linspace(0, 1, 5)

    sweep = ArraySweep(_param, array)
    assert isinstance(sweep.param, _BaseParameter)

    sweep_2 = ArraySweep(_param_complex, array)
    assert isinstance(sweep_2.param, _BaseParameter)

    arrayparam = ArraySetPointParam(name="arrayparam")
    sweep_3 = ArraySweep(arrayparam, array)
    assert isinstance(sweep_3.param, _BaseParameter)


def test_dond_explicit_exp_meas_sample(_param, experiment):
    experiment_2 = new_experiment("new-exp", "no-sample")

    data1 = dond(_param, do_plot=False, exp=experiment)
    assert data1[0].exp_name == "test-experiment"
    data2 = dond(_param, do_plot=False, exp=experiment_2, measurement_name="Meas")
    assert data2[0].name == "Meas"
    assert data2[0].sample_name == "no-sample"
    assert data2[0].exp_name == "new-exp"
    # by default the last experiment is used
    data3 = dond(_param, do_plot=False)
    assert data3[0].exp_name == "new-exp"


def test_dond_multi_datasets_explicit_exp_meas_sample(
    _param, _param_complex, experiment
):
    experiment_2 = new_experiment("new-exp", "no-sample")

    data1 = dond([_param], [_param_complex], do_plot=False, exp=experiment)
    assert data1[0][0].exp_name == "test-experiment"
    data2 = dond(
        [_param],
        [_param_complex],
        do_plot=False,
        exp=experiment_2,
        measurement_name="Meas",
    )
    assert data2[0][0].name == "Meas"
    assert data2[0][1].name == "Meas"
    assert data2[0][0].sample_name == "no-sample"
    assert data2[0][1].sample_name == "no-sample"
    assert data2[0][0].exp_name == "new-exp"
    assert data2[0][1].exp_name == "new-exp"
    # by default the last experiment is used
    data3 = dond([_param], [_param_complex], do_plot=False)
    assert data3[0][0].exp_name == "new-exp"
    assert data3[0][1].exp_name == "new-exp"


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize(
    "multiparamtype",
    [MultiSetPointParam, Multi2DSetPointParam, Multi2DSetPointParam2Sizes],
)
@given(n_points_pws=hst.integers(min_value=1, max_value=500))
@settings(deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_dond_0d_verify_shape(
    _param, _param_complex, multiparamtype, dummyinstrument, n_points_pws
):
    arrayparam = ArraySetPointParam(name="arrayparam")
    multiparam = multiparamtype(name="multiparam")
    paramwsetpoints = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_start(0)
    dummyinstrument.A.dummy_stop(1)
    dummyinstrument.A.dummy_n_points(n_points_pws)

    results = dond(
        arrayparam, multiparam, paramwsetpoints, _param, _param_complex, do_plot=False
    )
    expected_shapes = {}
    for i, name in enumerate(multiparam.full_names):
        expected_shapes[name] = tuple(multiparam.shapes[i])
    expected_shapes["arrayparam"] = tuple(arrayparam.shape)
    expected_shapes["simple_parameter"] = (1,)
    expected_shapes["simple_complex_parameter"] = (1,)
    expected_shapes[paramwsetpoints.full_name] = (n_points_pws,)

    ds = results[0]

    assert ds.description.shapes == expected_shapes

    data = ds.get_parameter_data()

    for name, data in data.items():
        for param_data in data.values():
            assert param_data.shape == expected_shapes[name]


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_0d_output_data(_param):
    exp = dond(_param)
    data = exp[0]
    assert data.parameters == _param.name
    loaded_data = data.get_parameter_data()["simple_parameter"]["simple_parameter"]
    assert loaded_data == np.array([_param.get()])


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_0d_output_type(_param, _param_complex, _param_callable):
    data_1 = do0d(_param)
    assert isinstance(data_1[0], DataSet) is True

    data_2 = do0d(_param_complex)
    assert isinstance(data_2[0], DataSet) is True

    data_3 = do0d(_param_callable)
    assert isinstance(data_3[0], DataSet) is True


@pytest.mark.usefixtures("experiment")
def test_dond_0d_parameter_with_array_vals():
    param = ArrayshapedParam(name="paramwitharrayval", vals=Arrays(shape=(10,)))
    results = dond(param)
    expected_shapes = {"paramwitharrayval": (10,)}
    assert results[0].description.shapes == expected_shapes


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("period", [None, 1])
@pytest.mark.parametrize("plot", [None, True, False])
@pytest.mark.parametrize("plot_config", [None, True, False])
def test_dond_0d_with_real_parameter(period, plot, plot_config):
    arrayparam = ArraySetPointParam(name="arrayparam")

    if plot_config is not None:
        config.dataset.dond_plot = plot_config

    output = dond(arrayparam, write_period=period, do_plot=plot)
    assert len(output[1]) == 1
    if plot is True or plot is None and plot_config is True:
        assert isinstance(output[1][0], matplotlib.axes.Axes)
    else:
        assert output[1][0] is None


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_0d_parameter_with_setpoints_2d(dummyinstrument):
    dummyinstrument.A.dummy_start(0)
    dummyinstrument.A.dummy_stop(10)
    dummyinstrument.A.dummy_n_points(10)
    dummyinstrument.A.dummy_start_2(2)
    dummyinstrument.A.dummy_stop_2(7)
    dummyinstrument.A.dummy_n_points_2(3)
    dataset, _, _ = dond(dummyinstrument.A.dummy_parameter_with_setpoints_2d)

    data = dataset.cache.data()[
        "dummyinstrument_ChanA_dummy_parameter_with_setpoints_2d"
    ]
    for array in data.values():
        assert array.shape == (10, 3)


@pytest.mark.usefixtures("experiment")
def test_dond_1d_parameter_with_array_vals(_param_set):
    param = ArrayshapedParam(name="paramwitharrayval", vals=Arrays(shape=(10,)))
    start = 0
    stop = 1
    num_points = 15  # make param
    sweep = LinSweep(_param_set, start, stop, num_points)

    results = dond(sweep, param, do_plot=False)
    expected_shapes = {"paramwitharrayval": (num_points, 10)}

    ds = results[0]

    assert ds.description.shapes == expected_shapes

    data = ds.get_parameter_data()

    for name, data in data.items():
        for param_data in data.values():
            assert param_data.shape == expected_shapes[name]


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize(
    "multiparamtype",
    [MultiSetPointParam, Multi2DSetPointParam, Multi2DSetPointParam2Sizes],
)
@given(
    num_points=hst.integers(min_value=1, max_value=5),
    n_points_pws=hst.integers(min_value=1, max_value=500),
)
@settings(deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_dond_1d_verify_shape(
    _param,
    _param_complex,
    _param_set,
    multiparamtype,
    dummyinstrument,
    num_points,
    n_points_pws,
):
    arrayparam = ArraySetPointParam(name="arrayparam")
    multiparam = multiparamtype(name="multiparam")
    paramwsetpoints = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_start(0)
    dummyinstrument.A.dummy_stop(1)
    dummyinstrument.A.dummy_n_points(n_points_pws)

    start = 0
    stop = 1
    delay = 0

    sweep_1 = LinSweep(_param_set, start, stop, num_points, delay)

    results = dond(
        sweep_1,
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


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_1d_additional_setpoints(_param, _param_complex, _param_set):
    additional_setpoints = [
        Parameter(f"additional_setter_parameter_{i}", set_cmd=None, get_cmd=None)
        for i in range(2)
    ]
    sweep = LinSweep(_param_set, 0, 1, 5, 0)

    for x in range(3):
        for y in range(4):
            additional_setpoints[0](x)
            additional_setpoints[1](y)
            results = dond(
                sweep, _param, _param_complex, additional_setpoints=additional_setpoints
            )
            for deps in results[0].description.interdeps.dependencies.values():
                assert len(deps) == 1 + len(additional_setpoints)
            # Calling the fixture won't work here due to loop-scope.
            # Thus, we make an explicit call to close plots. This will be
            # repeated in similarly design tests.
            plt.close("all")


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(num_points_p1=hst.integers(min_value=1, max_value=5))
@pytest.mark.usefixtures("experiment")
def test_dond_1d_additional_setpoints_shape(_param, _param_set, num_points_p1):
    arrayparam = ArraySetPointParam(name="arrayparam")
    array_shape = arrayparam.shape
    additional_setpoints = [
        Parameter(f"additional_setter_parameter_{i}", set_cmd=None, get_cmd=None)
        for i in range(2)
    ]

    sweep = LinSweep(_param_set, 0, 0.5, num_points_p1, 0)

    x = 1
    y = 2
    additional_setpoints[0](x)
    additional_setpoints[1](y)

    results = dond(
        sweep,
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
@pytest.mark.parametrize("plot", [None, True, False])
@pytest.mark.parametrize("plot_config", [None, True, False])
def test_dond_1d_plot(_param_set, _param, plot, plot_config):

    if plot_config is not None:
        config.dataset.dond_plot = plot_config

    sweep = LinSweep(_param_set, 0, 1, 1, 0)

    output = dond(sweep, _param, do_plot=plot)
    assert len(output[1]) == 1
    if plot is True or plot is None and plot_config is True:
        assert isinstance(output[1][0], matplotlib.axes.Axes)
    else:
        assert output[1][0] is None


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_1d_output_data(_param, _param_complex, _param_set):

    sweep_1 = LinSweep(_param_set, 0, 0.5, 5, 0)

    exp_1 = dond(sweep_1, _param, _param_complex)
    data_1 = exp_1[0]
    assert data_1.parameters == (
        f"{_param_set.name},{_param.name}," f"{_param_complex.name}"
    )
    loaded_data_1 = data_1.get_parameter_data()

    np.testing.assert_array_equal(
        loaded_data_1[_param.name][_param.name], np.ones(sweep_1._num_points)
    )
    np.testing.assert_array_equal(
        loaded_data_1[_param_complex.name][_param_complex.name],
        (1 + 1j) * np.ones(sweep_1._num_points),
    )
    np.testing.assert_array_equal(
        loaded_data_1[_param_complex.name][_param_set.name],
        np.linspace(sweep_1._start, sweep_1._stop, sweep_1._num_points),
    )
    np.testing.assert_array_equal(
        loaded_data_1[_param.name][_param_set.name],
        np.linspace(sweep_1._start, sweep_1._stop, sweep_1._num_points),
    )


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_1d_output_type(_param, _param_complex, _param_set):

    sweep_1 = LinSweep(_param_set, 0, 0.5, 2, 0)

    data_1 = dond(sweep_1, _param, _param_complex)
    assert isinstance(data_1[0], DataSet) is True


@pytest.mark.usefixtures("experiment")
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
def test_dond_2d_verify_shape(
    _param,
    _param_complex,
    _param_set,
    _param_set_2,
    multiparamtype,
    dummyinstrument,
    num_points_p1,
    num_points_p2,
    n_points_pws,
):
    arrayparam = ArraySetPointParam(name="arrayparam")
    multiparam = multiparamtype(name="multiparam")
    paramwsetpoints = dummyinstrument.A.dummy_parameter_with_setpoints
    dummyinstrument.A.dummy_start(0)
    dummyinstrument.A.dummy_stop(1)
    dummyinstrument.A.dummy_n_points(n_points_pws)

    start_p1 = 0
    stop_p1 = 1
    delay_p1 = 0
    sweep_1 = LinSweep(_param_set, start_p1, stop_p1, num_points_p1, delay_p1)
    start_p2 = 0.1
    stop_p2 = 1.1
    delay_p2 = 0
    sweep_2 = LinSweep(_param_set_2, start_p2, stop_p2, num_points_p2, delay_p2)

    results = dond(
        sweep_1,
        sweep_2,
        arrayparam,
        multiparam,
        paramwsetpoints,
        _param,
        _param_complex,
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

    for name, data in data.items():
        for param_data in data.values():
            assert param_data.shape == expected_shapes[name]


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_2d_additional_setpoints(_param, _param_complex, _param_set, _param_set_2):
    additional_setpoints = [
        Parameter(f"additional_setter_parameter_{i}", set_cmd=None, get_cmd=None)
        for i in range(2)
    ]

    start_p1 = 0
    stop_p1 = 0.5
    num_points_p1 = 5
    delay_p1 = 0
    sweep_1 = LinSweep(_param_set, start_p1, stop_p1, num_points_p1, delay_p1)
    start_p2 = 0.5
    stop_p2 = 1
    num_points_p2 = 5
    delay_p2 = 0.0
    sweep_2 = LinSweep(_param_set_2, start_p2, stop_p2, num_points_p2, delay_p2)

    for x in range(2):
        for y in range(3):
            additional_setpoints[0](x)
            additional_setpoints[1](y)
            results = dond(
                sweep_1,
                sweep_2,
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
def test_dond_2d_additional_setpoints_shape(
    _param, _param_complex, _param_set, _param_set_2, num_points_p1, num_points_p2
):
    arrayparam = ArraySetPointParam(name="arrayparam")
    array_shape = arrayparam.shape
    additional_setpoints = [
        Parameter(f"additional_setter_parameter_{i}", set_cmd=None, get_cmd=None)
        for i in range(2)
    ]

    start_p1 = 0
    stop_p1 = 1
    delay_p1 = 0
    sweep_1 = LinSweep(_param_set, start_p1, stop_p1, num_points_p1, delay_p1)
    start_p2 = 0.1
    stop_p2 = 1.1
    delay_p2 = 0
    sweep_2 = LinSweep(_param_set_2, start_p2, stop_p2, num_points_p2, delay_p2)

    x = 1
    y = 2
    additional_setpoints[0](x)
    additional_setpoints[1](y)

    results = dond(
        sweep_1,
        sweep_2,
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


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("plot", [None, True, False])
@pytest.mark.parametrize("plot_config", [None, True, False])
def test_dond_2d_plot(_param_set, _param_set_2, _param, plot, plot_config):

    if plot_config is not None:
        config.dataset.dond_plot = plot_config

    sweep_1 = LinSweep(_param_set, 0, 1, 1, 0)
    sweep_2 = LinSweep(_param_set_2, 0.1, 1.1, 2, 0)

    output = dond(sweep_1, sweep_2, _param, do_plot=plot)

    assert len(output[1]) == 1
    if plot is True or plot is None and plot_config is True:
        assert isinstance(output[1][0], matplotlib.axes.Axes)
    else:
        assert output[1][0] is None


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_2d_output_type(_param, _param_complex, _param_set, _param_set_2):

    sweep_1 = LinSweep(_param_set, 0, 0.5, 2, 0)
    sweep_2 = LinSweep(_param_set_2, 0.5, 1, 2, 0)

    data_1 = dond(sweep_1, sweep_2, _param, _param_complex)
    assert isinstance(data_1[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_2d_output_data(_param, _param_complex, _param_set, _param_set_2):
    sweep_1 = LinSweep(_param_set, 0, 0.5, 5, 0)
    sweep_2 = LinSweep(_param_set_2, 0.5, 1, 5, 0)
    exp_2 = dond(sweep_1, sweep_2, _param, _param_complex)
    data_2 = exp_2[0]
    assert data_2.parameters == (
        f"{_param_set.name},{_param_set_2.name}," f"{_param.name},{_param_complex.name}"
    )
    loaded_data_2 = data_2.get_parameter_data()
    expected_data_2 = np.ones(25).reshape(sweep_1._num_points, sweep_2._num_points)

    np.testing.assert_array_equal(
        loaded_data_2[_param.name][_param.name], expected_data_2
    )
    expected_data_3 = (1 + 1j) * np.ones(25).reshape(
        sweep_1._num_points, sweep_2._num_points
    )
    np.testing.assert_array_equal(
        loaded_data_2[_param_complex.name][_param_complex.name], expected_data_3
    )

    expected_setpoints_1 = np.repeat(
        np.linspace(sweep_1._start, sweep_1._stop, sweep_1._num_points),
        sweep_2._num_points,
    ).reshape(sweep_1._num_points, sweep_2._num_points)
    np.testing.assert_array_equal(
        loaded_data_2[_param_complex.name][_param_set.name], expected_setpoints_1
    )

    expected_setpoints_2 = np.tile(
        np.linspace(sweep_2._start, sweep_2._stop, sweep_2._num_points),
        sweep_1._num_points,
    ).reshape(sweep_1._num_points, sweep_2._num_points)
    np.testing.assert_array_equal(
        loaded_data_2[_param_complex.name][_param_set_2.name], expected_setpoints_2
    )


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_2d_multi_datasets_output_type(
    _param, _param_complex, _param_set, _param_set_2
):

    sweep_1 = LinSweep(_param_set, 0, 0.5, 2, 0)
    sweep_2 = LinSweep(_param_set_2, 0.5, 1, 2, 0)

    data_1 = dond(sweep_1, sweep_2, [_param], [_param_complex])
    assert isinstance(data_1[0][0], DataSet) is True
    assert isinstance(data_1[0][1], DataSet) is True


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("plot", [None, True, False])
@pytest.mark.parametrize("plot_config", [None, True, False])
def test_dond_2d_multiple_datasets_plot(
    _param_set, _param_set_2, _param, _param_2, plot, plot_config
):

    if plot_config is not None:
        config.dataset.dond_plot = plot_config

    sweep_1 = LinSweep(_param_set, 0, 1, 1, 0)
    sweep_2 = LinSweep(_param_set_2, 0.1, 1.1, 2, 0)

    output = dond(sweep_1, sweep_2, [_param], [_param_2], do_plot=plot)

    assert len(output[1]) == 2
    assert len(output[1][0]) == 1
    assert len(output[1][1]) == 1
    if plot is True or plot is None and plot_config is True:
        assert isinstance(output[1][0][0], matplotlib.axes.Axes)
        assert isinstance(output[1][1][0], matplotlib.axes.Axes)
    else:
        assert output[1][0][0] is None
        assert output[1][1][0] is None


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_2d_multi_datasets_with_callable_output_data(
    _param,
    _param_2,
    _param_complex,
    _param_complex_2,
    _param_set,
    _param_set_2,
    _string_callable,
):
    sweep_1 = LinSweep(_param_set, 0, 0.5, 5, 0)
    sweep_2 = LinSweep(_param_set_2, 0.5, 1, 5, 0)
    exp_1 = dond(
        sweep_1,
        sweep_2,
        [_string_callable, _param, _param_complex],
        [_string_callable, _param_2, _param_complex_2],
    )
    data_1 = exp_1[0][0]
    data_2 = exp_1[0][1]

    assert data_1.parameters == (
        f"{_param_set.name},{_param_set_2.name}," f"{_param.name},{_param_complex.name}"
    )
    assert data_2.parameters == (
        f"{_param_set.name},{_param_set_2.name},"
        f"{_param_2.name},{_param_complex_2.name}"
    )
    loaded_data_1 = data_1.get_parameter_data()
    expected_data_1_1 = np.ones(25).reshape(sweep_1._num_points, sweep_2._num_points)

    np.testing.assert_array_equal(
        loaded_data_1[_param.name][_param.name], expected_data_1_1
    )
    expected_data_1_2 = (1 + 1j) * np.ones(25).reshape(
        sweep_1._num_points, sweep_2._num_points
    )
    np.testing.assert_array_equal(
        loaded_data_1[_param_complex.name][_param_complex.name], expected_data_1_2
    )

    loaded_data_2 = data_2.get_parameter_data()
    expected_data_2_1 = 2 * np.ones(25).reshape(
        sweep_1._num_points, sweep_2._num_points
    )

    np.testing.assert_array_equal(
        loaded_data_2[_param_2.name][_param_2.name], expected_data_2_1
    )
    expected_data_2_2 = (2 + 2j) * np.ones(25).reshape(
        sweep_1._num_points, sweep_2._num_points
    )
    np.testing.assert_array_equal(
        loaded_data_2[_param_complex_2.name][_param_complex_2.name], expected_data_2_2
    )

    expected_setpoints_1 = np.repeat(
        np.linspace(sweep_1._start, sweep_1._stop, sweep_1._num_points),
        sweep_2._num_points,
    ).reshape(sweep_1._num_points, sweep_2._num_points)

    expected_setpoints_2 = np.tile(
        np.linspace(sweep_2._start, sweep_2._stop, sweep_2._num_points),
        sweep_1._num_points,
    ).reshape(sweep_1._num_points, sweep_2._num_points)

    np.testing.assert_array_equal(
        loaded_data_1[_param_complex.name][_param_set.name], expected_setpoints_1
    )
    np.testing.assert_array_equal(
        loaded_data_1[_param_complex.name][_param_set_2.name], expected_setpoints_2
    )
    np.testing.assert_array_equal(
        loaded_data_2[_param_complex_2.name][_param_set.name], expected_setpoints_1
    )
    np.testing.assert_array_equal(
        loaded_data_2[_param_complex_2.name][_param_set_2.name], expected_setpoints_2
    )
    np.testing.assert_array_equal(
        loaded_data_1[_param.name][_param_set.name], expected_setpoints_1
    )
    np.testing.assert_array_equal(
        loaded_data_1[_param.name][_param_set_2.name], expected_setpoints_2
    )
    np.testing.assert_array_equal(
        loaded_data_2[_param_2.name][_param_set.name], expected_setpoints_1
    )
    np.testing.assert_array_equal(
        loaded_data_2[_param_2.name][_param_set_2.name], expected_setpoints_2
    )
