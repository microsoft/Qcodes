import hypothesis.strategies as hst
import matplotlib
import matplotlib.axes
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from qcodes import config, validators
from qcodes.dataset import do0d, new_experiment
from qcodes.dataset.data_set import DataSet
from qcodes.instrument_drivers.mock_instruments import (
    ArraySetPointParam,
    Multi2DSetPointParam,
    Multi2DSetPointParam2Sizes,
    MultiSetPointParam,
)
from tests.dataset.conftest import ArrayshapedParam


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("period", [None, 1])
@pytest.mark.parametrize("plot", [None, True, False])
@pytest.mark.parametrize("plot_config", [True, None, False])
def test_do0d_with_real_parameter(period, plot, plot_config) -> None:
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
def test_do0d_with_complex_parameter(_param_complex, period, plot) -> None:
    do0d(_param_complex, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize(
    "period, plot", [(None, True), (None, False), (1, True), (1, False)]
)
def test_do0d_with_a_callable(_param_callable, period, plot) -> None:
    do0d(_param_callable, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize(
    "period, plot", [(None, True), (None, False), (1, True), (1, False)]
)
def test_do0d_with_2_parameters(_param, _param_complex, period, plot) -> None:
    do0d(_param, _param_complex, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize(
    "period, plot", [(None, True), (None, False), (1, True), (1, False)]
)
def test_do0d_with_parameter_and_a_callable(
    _param_complex, _param_callable, period, plot
) -> None:
    do0d(_param_callable, _param_complex, write_period=period, do_plot=plot)


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do0d_output_type_real_parameter(_param) -> None:
    data = do0d(_param)
    assert isinstance(data[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do0d_output_type_complex_parameter(_param_complex) -> None:
    data_complex = do0d(_param_complex)
    assert isinstance(data_complex[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do0d_output_type_callable(_param_callable) -> None:
    data_func = do0d(_param_callable)
    assert isinstance(data_func[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "experiment")
def test_do0d_output_data(_param) -> None:
    exp = do0d(_param)
    data = exp[0]
    assert data.description.interdeps.names == (_param.name,)
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
) -> None:
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

    for name, data_inner in data.items():
        for param_data in data_inner.values():
            assert param_data.shape == expected_shapes[name]


@pytest.mark.usefixtures("experiment")
def test_do0d_parameter_with_array_vals() -> None:
    param = ArrayshapedParam(
        name="paramwitharrayval", vals=validators.Arrays(shape=(10,))
    )
    results = do0d(param)
    expected_shapes = {"paramwitharrayval": (10,)}
    assert results[0].description.shapes == expected_shapes


def test_do0d_explicit_experiment(_param, experiment) -> None:
    experiment_2 = new_experiment("new-exp", "no-sample")

    data1 = do0d(_param, do_plot=False, exp=experiment)
    assert data1[0].exp_name == "test-experiment"
    data2 = do0d(_param, do_plot=False, exp=experiment_2)
    assert data2[0].exp_name == "new-exp"
    # by default the last experiment is used
    data3 = do0d(_param, do_plot=False)
    assert data3[0].exp_name == "new-exp"


@pytest.mark.usefixtures("experiment")
def test_do0d_explicit_name(_param) -> None:
    data1 = do0d(_param, do_plot=False, measurement_name="my measurement")
    assert data1[0].name == "my measurement"


@pytest.mark.usefixtures("experiment")
def test_do0d_parameter_with_setpoints_2d(dummyinstrument) -> None:
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


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_0d_output_type(_param, _param_complex, _param_callable) -> None:
    data_1 = do0d(_param)
    assert isinstance(data_1[0], DataSet) is True

    data_2 = do0d(_param_complex)
    assert isinstance(data_2[0], DataSet) is True

    data_3 = do0d(_param_callable)
    assert isinstance(data_3[0], DataSet) is True
