"""
These are the basic black box tests for the doNd functions.
"""

import logging
import re

import hypothesis.strategies as hst
import matplotlib
import matplotlib.axes
import matplotlib.colorbar
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from pytest import FixtureRequest, LogCaptureFixture
from typing_extensions import assert_type

import qcodes as qc
from qcodes import config, validators
from qcodes.dataset import (
    ArraySweep,
    DataSetProtocol,
    LinSweep,
    LogSweep,
    TogetherSweep,
    dond,
    new_experiment,
)
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.dond.do_nd import _Sweeper
from qcodes.instrument_drivers.mock_instruments import (
    ArraySetPointParam,
    Multi2DSetPointParam,
    Multi2DSetPointParam2Sizes,
    MultiSetPointParam,
)
from qcodes.parameters import ManualParameter, Parameter, ParameterBase
from qcodes.validators import Ints
from tests.dataset.conftest import ArrayshapedParam


class TrackingParameter(Parameter):
    """Parameter that keeps track of number of get and set operations"""

    def __init__(self, *args, **kwargs):
        self.set_count = 0
        self.get_count = 0
        super().__init__(*args, **kwargs)

    def set_raw(self, value):
        self.set_count += 1
        self.cache._set_from_raw_value(value)

    def get_raw(self):
        self.get_count += 1
        return self.cache.raw_value

    def reset_count(self) -> None:
        self.get_count = 0
        self.set_count = 0


class GetReturnsCountParameter(Parameter):
    """Parameter that keeps track of number of get and set operations
    Allows you to set a value but returns the get count rather
    than the value"""

    def __init__(self, *args, **kwargs):
        self.set_count = 0
        self.get_count = 0
        super().__init__(*args, **kwargs)

    def set_raw(self, value):
        self.set_count += 1
        self.cache._set_from_raw_value(value)

    def get_raw(self):
        self.get_count += 1
        return self.get_count

    def reset_count(self) -> None:
        self.get_count = 0
        self.set_count = 0


def test_linear_sweep_get_setpoints(_param) -> None:
    start = 0
    stop = 1
    num_points = 5
    delay = 1
    sweep = LinSweep(_param, start, stop, num_points, delay)

    np.testing.assert_array_equal(
        sweep.get_setpoints(), np.linspace(start, stop, num_points)
    )


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("cache_config", [True, False])
@pytest.mark.parametrize("cache_setting", [True, False, None])
def test_cache_config(_param, _param_2, cache_config, cache_setting) -> None:
    qc.config.dataset.in_memory_cache = cache_config
    start = 0
    stop = 1
    num_points = 5
    delay = 0
    sweep = LinSweep(_param, start, stop, num_points, delay)

    ds, _, _ = dond(sweep, _param_2, in_memory_cache=cache_setting)
    assert isinstance(ds, DataSet)

    if (cache_config and cache_setting is None) or cache_setting is True:
        assert ds.cache.live is True
    else:
        assert ds.cache.live is None


def test_linear_sweep_properties(_param, _param_complex) -> None:
    start = 0
    stop = 1
    num_points = 5
    delay = 1
    sweep = LinSweep(_param, start, stop, num_points, delay)
    assert isinstance(sweep.param, ParameterBase)
    assert sweep.delay == delay
    assert sweep.param == _param
    assert sweep.num_points == num_points

    # test default delay 0 with complex param
    sweep_2 = LinSweep(_param_complex, start, stop, num_points)
    assert sweep_2.delay == 0


def test_linear_sweep_parameter_class(_param, _param_complex) -> None:
    start = 0
    stop = 1
    num_points = 5
    delay = 1
    sweep = LinSweep(_param, start, stop, num_points, delay)
    assert isinstance(sweep.param, ParameterBase)

    sweep_2 = LinSweep(_param_complex, start, stop, num_points)
    assert isinstance(sweep_2.param, ParameterBase)

    arrayparam = ArraySetPointParam(name="arrayparam")
    sweep_3 = LinSweep(arrayparam, start, stop, num_points)
    assert isinstance(sweep_3.param, ParameterBase)


def test_log_sweep_get_setpoints(_param) -> None:
    start = 0
    stop = 1
    num_points = 5
    delay = 1
    sweep = LogSweep(_param, start, stop, num_points, delay)

    np.testing.assert_array_equal(
        sweep.get_setpoints(), np.logspace(start, stop, num_points)
    )


def test_log_sweep_properties(_param, _param_complex) -> None:
    start = 0
    stop = 1
    num_points = 5
    delay = 1
    sweep = LogSweep(_param, start, stop, num_points, delay)
    assert isinstance(sweep.param, ParameterBase)
    assert sweep.delay == delay
    assert sweep.param == _param
    assert sweep.num_points == num_points

    # test default delay 0 with complex param
    sweep_2 = LogSweep(_param_complex, start, stop, num_points)
    assert sweep_2.delay == 0


def test_log_sweep_parameter_class(_param, _param_complex) -> None:
    start = 0
    stop = 1
    num_points = 5
    delay = 1
    sweep = LogSweep(_param, start, stop, num_points, delay)
    assert isinstance(sweep.param, ParameterBase)

    sweep_2 = LogSweep(_param_complex, start, stop, num_points)
    assert isinstance(sweep_2.param, ParameterBase)

    arrayparam = ArraySetPointParam(name="arrayparam")
    sweep_3 = LogSweep(arrayparam, start, stop, num_points)
    assert isinstance(sweep_3.param, ParameterBase)


def test_array_sweep_get_setpoints(_param) -> None:
    array = np.linspace(0, 1, 5)
    delay = 1
    sweep = ArraySweep(_param, array, delay)

    np.testing.assert_array_equal(sweep.get_setpoints(), array)

    array2 = [1, 2, 3, 4, 5, 5.2]
    sweep2: ArraySweep[np.floating] = ArraySweep(_param, array2)

    np.testing.assert_array_equal(sweep2.get_setpoints(), np.array(array2))


def test_array_sweep_properties(_param) -> None:
    array = np.linspace(0, 1, 5)
    delay = 1
    sweep = ArraySweep(_param, array, delay)
    assert isinstance(sweep.param, ParameterBase)
    assert sweep.delay == delay
    assert sweep.param == _param
    assert sweep.num_points == len(array)

    # test default delay 0
    sweep_2 = ArraySweep(_param, array)
    assert sweep_2.delay == 0


def test_array_sweep_parameter_class(_param, _param_complex) -> None:
    array = np.linspace(0, 1, 5)

    sweep = ArraySweep(_param, array)
    assert isinstance(sweep.param, ParameterBase)

    sweep_2 = ArraySweep(_param_complex, array)
    assert isinstance(sweep_2.param, ParameterBase)

    arrayparam = ArraySetPointParam(name="arrayparam")
    sweep_3 = ArraySweep(arrayparam, array)
    assert isinstance(sweep_3.param, ParameterBase)


def test_dond_explicit_exp_meas_sample(_param, experiment) -> None:
    experiment_2 = new_experiment("new-exp", "no-sample")

    data1 = dond(_param, do_plot=False, exp=experiment)
    dataset1 = data1[0]
    assert isinstance(dataset1, DataSetProtocol)
    assert dataset1.exp_name == "test-experiment"
    assert dataset1.name == "results"
    assert dataset1.sample_name == "test-sample"
    data2 = dond(_param, do_plot=False, exp=experiment_2, measurement_name="Meas")
    dataset2 = data2[0]
    assert isinstance(dataset2, DataSetProtocol)
    assert dataset2.name == "Meas"
    assert dataset2.sample_name == "no-sample"
    assert dataset2.exp_name == "new-exp"
    # by default the last experiment is used
    data3 = dond(_param, do_plot=False)
    dataset3 = data3[0]
    assert isinstance(dataset3, DataSetProtocol)
    assert dataset3.exp_name == "new-exp"
    assert dataset3.sample_name == "no-sample"
    assert dataset3.name == "results"


def test_dond_multi_datasets_explicit_exp_meas_sample(
    _param, _param_complex, experiment
) -> None:
    experiment_2 = new_experiment("new-exp", "no-sample")

    data1 = dond([_param], [_param_complex], do_plot=False, exp=experiment)
    datasets1 = data1[0]
    assert isinstance(datasets1, tuple)
    assert datasets1[0].exp_name == "test-experiment"
    data2 = dond(
        [_param],
        [_param_complex],
        do_plot=False,
        exp=experiment_2,
        measurement_name="Meas",
    )
    datasets2 = data2[0]
    assert isinstance(datasets2, tuple)
    assert datasets2[0].name == "Meas"
    assert datasets2[1].name == "Meas"
    assert datasets2[0].sample_name == "no-sample"
    assert datasets2[1].sample_name == "no-sample"
    assert datasets2[0].exp_name == "new-exp"
    assert datasets2[1].exp_name == "new-exp"
    # by default the last experiment is used
    data3 = dond([_param], [_param_complex], do_plot=False)
    datasets3 = data3[0]
    assert isinstance(datasets3, tuple)
    assert datasets3[0].exp_name == "new-exp"
    assert datasets3[1].exp_name == "new-exp"


def test_dond_multi_datasets_explicit_meas_names(
    _param, _param_complex, experiment
) -> None:
    data1 = dond(
        [_param],
        [_param_complex],
        measurement_name=["foo", "bar"],
        do_plot=False,
        exp=experiment,
    )
    datasets1 = data1[0]
    assert isinstance(datasets1, tuple)
    assert datasets1[0].name == "foo"
    assert datasets1[1].name == "bar"


def test_dond_multi_datasets_meas_names_len_mismatch(_param, experiment) -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("Got 2 measurement names but should create 1 dataset(s)."),
    ):
        dond(
            [_param],
            measurement_name=["foo", "bar"],
            do_plot=False,
            exp=experiment,
        )


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize(
    "multiparamtype",
    [MultiSetPointParam, Multi2DSetPointParam, Multi2DSetPointParam2Sizes],
)
@given(n_points_pws=hst.integers(min_value=1, max_value=500))
@settings(deadline=None, suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_dond_0d_verify_shape(
    _param, _param_complex, multiparamtype, dummyinstrument, n_points_pws
) -> None:
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
    assert isinstance(ds, DataSetProtocol)
    assert ds.description.shapes == expected_shapes

    data = ds.get_parameter_data()

    for name, data_inner in data.items():
        for param_data in data_inner.values():
            assert param_data.shape == expected_shapes[name]


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_0d_output_data(_param) -> None:
    exp = dond(_param)
    data = exp[0]
    assert isinstance(data, DataSetProtocol)
    assert data.description.interdeps.names == (_param.name,)
    loaded_data = data.get_parameter_data()["simple_parameter"]["simple_parameter"]
    assert loaded_data == np.array([_param.get()])


@pytest.mark.usefixtures("experiment")
def test_dond_0d_parameter_with_array_vals() -> None:
    param = ArrayshapedParam(
        name="paramwitharrayval", vals=validators.Arrays(shape=(10,))
    )
    results = dond(param)
    expected_shapes = {"paramwitharrayval": (10,)}
    dataset = results[0]
    assert isinstance(dataset, DataSetProtocol)
    assert dataset.description.shapes == expected_shapes


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("period", [None, 1])
@pytest.mark.parametrize("plot", [None, True, False])
@pytest.mark.parametrize("plot_config", [None, True, False])
def test_dond_0d_with_real_parameter(period, plot, plot_config) -> None:
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
def test_dond_0d_parameter_with_setpoints_2d(dummyinstrument) -> None:
    dummyinstrument.A.dummy_start(0)
    dummyinstrument.A.dummy_stop(10)
    dummyinstrument.A.dummy_n_points(10)
    dummyinstrument.A.dummy_start_2(2)
    dummyinstrument.A.dummy_stop_2(7)
    dummyinstrument.A.dummy_n_points_2(3)
    dataset, _, _ = dond(dummyinstrument.A.dummy_parameter_with_setpoints_2d)
    assert isinstance(dataset, DataSetProtocol)
    data = dataset.cache.data()[
        "dummyinstrument_ChanA_dummy_parameter_with_setpoints_2d"
    ]
    for array in data.values():
        assert array.shape == (10, 3)


@pytest.mark.usefixtures("experiment")
def test_dond_1d_parameter_with_array_vals(_param_set) -> None:
    param = ArrayshapedParam(
        name="paramwitharrayval", vals=validators.Arrays(shape=(10,))
    )
    start = 0
    stop = 1
    num_points = 15  # make param
    sweep = LinSweep(_param_set, start, stop, num_points)

    results = dond(sweep, param, do_plot=False)
    expected_shapes = {"paramwitharrayval": (num_points, 10)}

    ds = results[0]
    assert isinstance(ds, DataSetProtocol)
    assert ds.description.shapes == expected_shapes

    data = ds.get_parameter_data()

    for name, data_inner in data.items():
        for param_data in data_inner.values():
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
    dataset = results[0]
    assert isinstance(dataset, DataSetProtocol)
    assert dataset.description.shapes == expected_shapes


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_1d_additional_setpoints(_param, _param_complex, _param_set) -> None:
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
            dataset = results[0]
            assert isinstance(dataset, DataSetProtocol)
            for deps in dataset.description.interdeps.dependencies.values():
                assert len(deps) == 1 + len(additional_setpoints)
            # Calling the fixture won't work here due to loop-scope.
            # Thus, we make an explicit call to close plots. This will be
            # repeated in similarly design tests.
            plt.close("all")


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(num_points_p1=hst.integers(min_value=1, max_value=5))
@pytest.mark.usefixtures("experiment")
def test_dond_1d_additional_setpoints_shape(_param, _param_set, num_points_p1) -> None:
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
    dataset = results[0]
    assert isinstance(dataset, DataSetProtocol)
    assert dataset.description.shapes == expected_shapes


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("plot", [None, True, False])
@pytest.mark.parametrize("plot_config", [None, True, False])
def test_dond_1d_plot(_param_set, _param, plot, plot_config) -> None:
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
def test_dond_1d_output_data(_param, _param_complex, _param_set) -> None:
    sweep_1 = LinSweep(_param_set, 0, 0.5, 5, 0)

    exp_1 = dond(sweep_1, _param, _param_complex)
    data_1 = exp_1[0]
    assert isinstance(data_1, DataSetProtocol)
    assert set(data_1.description.interdeps.names) == {
        _param_set.name,
        _param.name,
        _param_complex.name,
    }
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
def test_dond_1d_output_type(_param, _param_complex, _param_set) -> None:
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
    ds = results[0]
    assert isinstance(ds, DataSetProtocol)

    assert ds.description.shapes == expected_shapes

    data = ds.get_parameter_data()

    for name, data_inner in data.items():
        for param_data in data_inner.values():
            assert param_data.shape == expected_shapes[name]


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_2d_additional_setpoints(
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
            dataset = results[0]
            assert isinstance(dataset, DataSetProtocol)
            for deps in dataset.description.interdeps.dependencies.values():
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
) -> None:
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
    dataset = results[0]
    assert isinstance(dataset, DataSetProtocol)
    assert dataset.description.shapes == expected_shapes


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("plot", [None, True, False])
@pytest.mark.parametrize("plot_config", [None, True, False])
def test_dond_2d_plot(_param_set, _param_set_2, _param, plot, plot_config) -> None:
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
def test_dond_2d_output_type(_param, _param_complex, _param_set, _param_set_2) -> None:
    sweep_1 = LinSweep(_param_set, 0, 0.5, 2, 0)
    sweep_2 = LinSweep(_param_set_2, 0.5, 1, 2, 0)

    data_1 = dond(sweep_1, sweep_2, _param, _param_complex)
    assert isinstance(data_1[0], DataSet) is True


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_2d_output_data(_param, _param_complex, _param_set, _param_set_2) -> None:
    sweep_1 = LinSweep(_param_set, 0, 0.5, 5, 0)
    sweep_2 = LinSweep(_param_set_2, 0.5, 1, 5, 0)
    exp_2 = dond(sweep_1, sweep_2, _param, _param_complex)
    data_2 = exp_2[0]
    assert isinstance(data_2, DataSetProtocol)
    assert set(data_2.description.interdeps.names) == {
        _param_set.name,
        _param_set_2.name,
        _param.name,
        _param_complex.name,
    }
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
) -> None:
    sweep_1 = LinSweep(_param_set, 0, 0.5, 2, 0)
    sweep_2 = LinSweep(_param_set_2, 0.5, 1, 2, 0)

    data_1 = dond(sweep_1, sweep_2, [_param], [_param_complex])
    datasets = data_1[0]
    assert isinstance(datasets, tuple)
    assert isinstance(datasets[0], DataSet) is True
    assert isinstance(datasets[1], DataSet) is True


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_2d_multi_datasets_multi_exp(
    _param, _param_complex, _param_set, _param_set_2, request: FixtureRequest
) -> None:
    exp1 = new_experiment("test-experiment-1", sample_name="test-sample-1")
    exp2 = new_experiment("test-experiment-2", sample_name="test-sample-2")

    request.addfinalizer(exp1.conn.close)
    request.addfinalizer(exp2.conn.close)

    sweep_1 = LinSweep(_param_set, 0, 0.5, 2, 0)
    sweep_2 = LinSweep(_param_set_2, 0.5, 1, 2, 0)

    data_1 = dond(sweep_1, sweep_2, [_param], [_param_complex], exp=[exp1, exp2])
    datasets = data_1[0]
    assert isinstance(datasets, tuple)
    assert isinstance(datasets[0], DataSet) is True
    assert isinstance(datasets[1], DataSet) is True

    assert datasets[0].exp_id == exp1.exp_id
    assert datasets[1].exp_id == exp2.exp_id


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_2d_multi_datasets_multi_exp_inconsistent_raises(
    _param, _param_complex, _param_set, _param_set_2, request: FixtureRequest
) -> None:
    exp1 = new_experiment("test-experiment-1", sample_name="test-sample-1")
    exp2 = new_experiment("test-experiment-2", sample_name="test-sample-2")
    exp3 = new_experiment("test-experiment-3", sample_name="test-sample-3")

    request.addfinalizer(exp1.conn.close)
    request.addfinalizer(exp2.conn.close)
    request.addfinalizer(exp3.conn.close)

    sweep_1 = LinSweep(_param_set, 0, 0.5, 2, 0)
    sweep_2 = LinSweep(_param_set_2, 0.5, 1, 2, 0)

    with pytest.raises(
        ValueError, match="Inconsistent number of datasets and experiments"
    ):
        dond(sweep_1, sweep_2, [_param], [_param_complex], exp=[exp1, exp2, exp3])


@pytest.mark.usefixtures("plot_close", "experiment")
@pytest.mark.parametrize("plot", [None, True, False])
@pytest.mark.parametrize("plot_config", [None, True, False])
def test_dond_2d_multiple_datasets_plot(
    _param_set, _param_set_2, _param, _param_2, plot, plot_config
) -> None:
    if plot_config is not None:
        config.dataset.dond_plot = plot_config

    sweep_1 = LinSweep(_param_set, 0, 1, 1, 0)
    sweep_2 = LinSweep(_param_set_2, 0.1, 1.1, 2, 0)

    output = dond(sweep_1, sweep_2, [_param], [_param_2], do_plot=plot)

    axes = output[1]

    assert isinstance(axes, tuple)
    assert len(axes) == 2
    assert isinstance(axes[0], tuple)
    assert len(axes[0]) == 1
    assert isinstance(axes[1], tuple)
    assert len(axes[1]) == 1
    if plot is True or plot is None and plot_config is True:
        assert isinstance(axes[0][0], matplotlib.axes.Axes)
        assert isinstance(axes[1][0], matplotlib.axes.Axes)
    else:
        assert axes[0][0] is None
        assert axes[1][0] is None


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_2d_multi_datasets_with_callable_output_data(
    _param,
    _param_2,
    _param_complex,
    _param_complex_2,
    _param_set,
    _param_set_2,
    _string_callable,
) -> None:
    sweep_1 = LinSweep(_param_set, 0, 0.5, 5, 0)
    sweep_2 = LinSweep(_param_set_2, 0.5, 1, 5, 0)
    exp_1 = dond(
        sweep_1,
        sweep_2,
        [_string_callable, _param, _param_complex],
        [_string_callable, _param_2, _param_complex_2],
    )
    datasets = exp_1[0]
    assert isinstance(datasets, tuple)
    data_1 = datasets[0]
    data_2 = datasets[1]

    assert set(data_1.description.interdeps.names) == {
        _param_set.name,
        _param_set_2.name,
        _param.name,
        _param_complex.name,
    }
    assert set(data_2.description.interdeps.names) == {
        _param_set.name,
        _param_set_2.name,
        _param_2.name,
        _param_complex_2.name,
    }
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


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep(_param_set, _param_set_2, _param, _param_2) -> None:
    sweep_1 = LinSweep(_param_set, 0, 1, 10, 0)
    sweep_2 = LinSweep(_param_set_2, 1, 2, 10, 0)

    together_sweep = TogetherSweep(sweep_1, sweep_2)

    output = dond(together_sweep, [_param], [_param_2], do_plot=False)

    assert sweep_1.param.name == "simple_setter_parameter"
    assert _param.name == "simple_parameter"

    assert sweep_2.param.name == "simple_setter_parameter_2"
    assert _param_2.name == "simple_parameter_2"
    datasets = output[0]
    assert isinstance(datasets, tuple)
    assert set(datasets[0].description.interdeps.names) == {
        "simple_setter_parameter",
        "simple_setter_parameter_2",
        "simple_parameter",
    }
    assert set(datasets[1].description.interdeps.names) == {
        "simple_setter_parameter",
        "simple_setter_parameter_2",
        "simple_parameter_2",
    }


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep_sweeper(_param_set, _param_set_2, _param) -> None:
    sweep_len = 10

    delay_1 = 0.1

    sweep_1 = LinSweep(_param_set, 0, 1, sweep_len, delay_1)

    delay_2 = 0.2

    sweep_2 = LinSweep(_param_set_2, 1, 2, sweep_len, delay_2)

    together_sweep = TogetherSweep(sweep_1, sweep_2)

    sweeper = _Sweeper([together_sweep], [])

    expected_sweeper_groups = (
        (sweep_1.param,),
        (sweep_2.param,),
        (
            sweep_1.param,
            sweep_2.param,
        ),
    )
    for g in expected_sweeper_groups:
        assert g in sweeper.sweep_groupes
    assert len(expected_sweeper_groups) == len(sweeper.sweep_groupes)
    assert sweeper.shape == (10,)
    assert sweeper.all_setpoint_params == (sweep_1.param, sweep_2.param)

    assert list(sweeper.setpoints_dict.keys()) == [
        "simple_setter_parameter",
        "simple_setter_parameter_2",
    ]
    assert sweeper.setpoints_dict["simple_setter_parameter"] == list(
        sweep_1.get_setpoints()
    )
    assert sweeper.setpoints_dict["simple_setter_parameter_2"] == list(
        sweep_2.get_setpoints()
    )

    for output, setpoint_1, setpoint_2 in zip(
        sweeper, sweep_1.get_setpoints(), sweep_2.get_setpoints()
    ):
        assert output[0].parameter == sweep_1.param
        assert output[0].new_value == setpoint_1
        assert output[0].should_set is True
        assert output[0].delay == delay_1

        assert output[1].parameter == sweep_2.param
        assert output[1].new_value == setpoint_2
        assert output[1].should_set is True
        assert output[1].delay == delay_2


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep_sweeper_combined() -> None:
    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    d = ManualParameter("d", initial_value=1)
    e = ManualParameter("e", initial_value=2)
    f = ManualParameter("f", initial_value=3)
    sweepA = LinSweep(a, 0, 3, 10)
    sweepB = LinSweep(b, 5, 7, 10)
    sweepC = LinSweep(c, 8, 12, 10)

    datasets, _, _ = dond(
        TogetherSweep(sweepA, sweepB),
        sweepC,
        d,
        e,
        f,
        do_plot=False,
        dataset_dependencies={
            "ds1": (a, c, d),
            "ds2": (b, c, e),
            "ds3": (b, c, f),
        },
    )
    assert isinstance(datasets, tuple)
    assert set(datasets[0].description.interdeps.names) == {"a", "c", "d"}
    assert datasets[0].name == "ds1"
    assert set(datasets[1].description.interdeps.names) == {"b", "c", "e"}
    assert datasets[1].name == "ds2"
    assert set(datasets[2].description.interdeps.names) == {"b", "c", "f"}
    assert datasets[2].name == "ds3"


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep_sweeper_combined_2_in_1() -> None:
    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    d = ManualParameter("d", initial_value=1)
    e = ManualParameter("e", initial_value=2)
    f = ManualParameter("f", initial_value=3)
    sweep_a = LinSweep(a, 0, 3, 10)
    sweep_b = LinSweep(b, 5, 7, 10)
    sweep_c = LinSweep(c, 8, 12, 10)

    datasets, _, _ = dond(
        TogetherSweep(sweep_a, sweep_b),
        sweep_c,
        d,
        e,
        f,
        do_plot=False,
        dataset_dependencies={
            "ds1": (a, c, d, f),
            "ds2": (b, c, e),
        },
    )
    assert isinstance(datasets, tuple)
    assert set(datasets[0].description.interdeps.names) == {"a", "c", "d", "f"}
    assert datasets[0].name == "ds1"
    assert set(datasets[1].description.interdeps.names) == {"b", "c", "e"}
    assert datasets[1].name == "ds2"


def test_dond_together_sweep_sweeper_mixed_splitting() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Measured parameters have been grouped both in input "
            "and given in dataset dependencies. This is not supported, "
            "group measurement parameters either in input or in dataset dependencies."
        ),
    ):
        a = ManualParameter("a", initial_value=0)
        b = ManualParameter("b", initial_value=0)
        c = ManualParameter("c", initial_value=0)
        d = ManualParameter("d", initial_value=1)
        e = ManualParameter("e", initial_value=2)
        f = ManualParameter("f", initial_value=3)
        sweepA = LinSweep(a, 0, 3, 10)
        sweepB = LinSweep(b, 5, 7, 10)
        sweepC = LinSweep(c, 8, 12, 10)

        datasets, _, _ = dond(
            TogetherSweep(sweepA, sweepB),
            sweepC,
            [d],
            [e],
            [f],
            do_plot=False,
            dataset_dependencies={
                "ds1": (a, c, d),
                "ds2": (b, c, e),
                "ds3": (b, c, f),
            },
        )


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep_sweeper_combined_explict_names() -> None:
    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    d = ManualParameter("d", initial_value=1)
    e = ManualParameter("e", initial_value=2)
    f = ManualParameter("f", initial_value=3)
    sweepA = LinSweep(a, 0, 3, 10)
    sweepB = LinSweep(b, 5, 7, 10)
    sweepC = LinSweep(c, 8, 12, 10)

    datasets, _, _ = dond(
        TogetherSweep(sweepA, sweepB),
        sweepC,
        d,
        e,
        f,
        measurement_name=("ds1", "ds2", "ds3"),
        do_plot=False,
        dataset_dependencies={
            "ds1": (a, c, d),
            "ds2": (b, c, e),
            "ds3": (b, c, f),
        },
    )
    assert isinstance(datasets, tuple)
    assert set(datasets[0].description.interdeps.names) == {"a", "c", "d"}
    assert datasets[0].name == "ds1"
    assert set(datasets[1].description.interdeps.names) == {"b", "c", "e"}
    assert datasets[1].name == "ds2"
    assert set(datasets[2].description.interdeps.names) == {"b", "c", "f"}
    assert datasets[2].name == "ds3"


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep_sweeper_combined_explict_names_inconsistent() -> None:
    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    d = ManualParameter("d", initial_value=1)
    e = ManualParameter("e", initial_value=2)
    f = ManualParameter("f", initial_value=3)
    sweepA = LinSweep(a, 0, 3, 10)
    sweepB = LinSweep(b, 5, 7, 10)
    sweepC = LinSweep(c, 8, 12, 10)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Inconsistent measurement names: measurement_name contains "
            "('ds1', 'ds2', 'ds4') but dataset_dependencies contains ('ds1', 'ds2', 'ds3')."
        ),
    ):
        datasets, _, _ = dond(
            TogetherSweep(sweepA, sweepB),
            sweepC,
            d,
            e,
            f,
            measurement_name=("ds1", "ds2", "ds4"),
            do_plot=False,
            dataset_dependencies={
                "ds1": (a, c, d),
                "ds2": (b, c, e),
                "ds3": (b, c, f),
            },
        )


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep_sweeper_combined_explict_names_and_single_name() -> None:
    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    d = ManualParameter("d", initial_value=1)
    e = ManualParameter("e", initial_value=2)
    f = ManualParameter("f", initial_value=3)
    sweepA = LinSweep(a, 0, 3, 10)
    sweepB = LinSweep(b, 5, 7, 10)
    sweepC = LinSweep(c, 8, 12, 10)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Creating multiple datasets but only one measurement name given."
        ),
    ):
        datasets, _, _ = dond(
            TogetherSweep(sweepA, sweepB),
            sweepC,
            d,
            e,
            f,
            measurement_name="my_measurement",
            do_plot=False,
            dataset_dependencies={
                "ds1": (a, c, d),
                "ds2": (b, c, e),
                "ds3": (b, c, f),
            },
        )


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep_sweeper_combined_lists() -> None:
    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    d = ManualParameter("d", initial_value=1)
    e = ManualParameter("e", initial_value=2)
    f = ManualParameter("f", initial_value=3)
    sweepA = LinSweep(a, 0, 3, 10)
    sweepB = LinSweep(b, 5, 7, 10)
    sweepC = LinSweep(c, 8, 12, 10)

    datasets, _, _ = dond(
        TogetherSweep(sweepA, sweepB),
        sweepC,
        d,
        e,
        f,
        do_plot=False,
        dataset_dependencies={
            "ds1": [a, c, d],
            "ds2": [b, c, e],
            "ds3": [b, c, f],
        },
    )
    assert isinstance(datasets, tuple)
    assert set(datasets[0].description.interdeps.names) == {"a", "c", "d"}
    assert datasets[0].name == "ds1"
    assert set(datasets[1].description.interdeps.names) == {"b", "c", "e"}
    assert datasets[1].name == "ds2"
    assert set(datasets[2].description.interdeps.names) == {"b", "c", "f"}
    assert datasets[2].name == "ds3"


@given(
    n_points_1=hst.integers(min_value=1, max_value=500),
    n_points_2=hst.integers(min_value=1, max_value=500),
)
def test_together_sweep_validation(n_points_1, n_points_2) -> None:
    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    sweepA = LinSweep(a, 0, 3, n_points_1)
    sweepB = LinSweep(b, 5, 7, n_points_2)

    if n_points_1 != n_points_2:
        with pytest.raises(
            ValueError, match="All Sweeps in a TogetherSweep must have the same length"
        ):
            TogetherSweep(sweepA, sweepB)


def test_empty_together_sweep_raises() -> None:
    with pytest.raises(
        ValueError, match="A TogetherSweep must contain at least one sweep."
    ):
        TogetherSweep()


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep_more_parameters() -> None:
    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    d = ManualParameter("d", initial_value=1)
    e = ManualParameter("e", initial_value=2)
    f = ManualParameter("f", initial_value=3)
    sweepA = LinSweep(a, 0, 3, 10)
    sweepB = LinSweep(b, 5, 7, 10)
    sweepC = LinSweep(c, 8, 12, 10)

    dataset, _, _ = dond(
        TogetherSweep(sweepA, sweepB),
        sweepC,
        d,
        e,
        f,
        do_plot=False,
    )
    assert isinstance(dataset, DataSetProtocol)
    assert set(dataset.description.interdeps.names) == {"a", "b", "c", "d", "e", "f"}


def test_dond_together_sweep_sweeper_combined_missing_in_dataset_dependencies() -> None:
    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    d = ManualParameter("d", initial_value=1)
    e = ManualParameter("e", initial_value=2)
    f = ManualParameter("f", initial_value=3)
    sweepA = LinSweep(a, 0, 3, 10)
    sweepB = LinSweep(b, 5, 7, 10)
    sweepC = LinSweep(c, 8, 12, 10)

    with pytest.raises(
        ValueError,
        match="Parameter f is measured but not added to any dataset",
    ):
        datasets, _, _ = dond(
            TogetherSweep(sweepA, sweepB),
            sweepC,
            d,
            e,
            f,
            do_plot=False,
            dataset_dependencies={
                "ds1": (a, c, d),
                "ds2": (b, c, e),
            },
        )


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep_sweeper_wrong_sp_in_dataset_dependencies() -> None:
    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    d = ManualParameter("d", initial_value=1)
    e = ManualParameter("e", initial_value=2)
    f = ManualParameter("f", initial_value=3)
    sweepA = LinSweep(a, 0, 3, 10)
    sweepB = LinSweep(b, 5, 7, 10)
    sweepC = LinSweep(c, 8, 12, 10)

    with pytest.raises(ValueError, match="not among the expected groups of setpoints"):
        datasets, _, _ = dond(
            TogetherSweep(sweepA, sweepB),
            sweepC,
            d,
            e,
            f,
            do_plot=False,
            dataset_dependencies={"ds1": (a, b, d), "ds2": (b, c, e), "ds3": (b, c, f)},
        )


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep_sweeper_wrong_mp_in_dataset_dependencies() -> None:
    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    d = ManualParameter("d", initial_value=1)
    e = ManualParameter("e", initial_value=2)
    f = ManualParameter("f", initial_value=3)
    g = ManualParameter("g", initial_value=4)
    sweepA = LinSweep(a, 0, 3, 10)
    sweepB = LinSweep(b, 5, 7, 10)
    sweepC = LinSweep(c, 8, 12, 10)

    with pytest.raises(
        ValueError,
        match="which is not among the expected groups of setpoints",
    ):
        datasets, _, _ = dond(
            TogetherSweep(sweepA, sweepB),
            sweepC,
            d,
            e,
            f,
            do_plot=False,
            dataset_dependencies={
                "ds1": (a, c, d),
                "ds2": (b, c, e),
                "ds3": (b, c, f),
                "ds4": (b, c, g),
            },
        )


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep_parameter_with_setpoints(dummyinstrument) -> None:
    outer_shape = 10
    inner_shape = 15

    n_points_a = 20
    dummyinstrument.A.dummy_n_points(n_points_a)
    n_points_b = 25
    dummyinstrument.B.dummy_n_points(n_points_b)

    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    sweep_a = LinSweep(a, 0, 3, outer_shape)
    sweep_b = LinSweep(b, 5, 7, outer_shape)
    sweep_c = LinSweep(c, 8, 12, inner_shape)

    datasets, _, _ = dond(
        TogetherSweep(sweep_a, sweep_b),
        sweep_c,
        [dummyinstrument.A.dummy_parameter_with_setpoints],
        [dummyinstrument.B.dummy_parameter_with_setpoints],
        do_plot=False,
    )
    assert isinstance(datasets, tuple)
    assert set(datasets[0].description.interdeps.names) == {
        "a",
        "b",
        "c",
        "dummyinstrument_ChanA_dummy_sp_axis",
        "dummyinstrument_ChanA_dummy_parameter_with_setpoints",
    }
    assert datasets[0].description.shapes is not None
    assert len(datasets[0].description.shapes) == 1
    assert datasets[0].description.shapes[
        "dummyinstrument_ChanA_dummy_parameter_with_setpoints"
    ] == (outer_shape, inner_shape, n_points_a)

    assert set(datasets[1].description.interdeps.names) == {
        "a",
        "b",
        "c",
        "dummyinstrument_ChanB_dummy_sp_axis",
        "dummyinstrument_ChanB_dummy_parameter_with_setpoints",
    }
    assert datasets[1].description.shapes is not None
    assert len(datasets[1].description.shapes) == 1
    assert datasets[1].description.shapes[
        "dummyinstrument_ChanB_dummy_parameter_with_setpoints"
    ] == (outer_shape, inner_shape, n_points_b)


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep_parameter_with_setpoints_explicit_mapping(
    dummyinstrument,
) -> None:
    outer_shape = 10
    inner_shape = 15

    n_points_a = 20
    dummyinstrument.A.dummy_n_points(n_points_a)
    n_points_b = 25
    dummyinstrument.B.dummy_n_points(n_points_b)

    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    sweep_a = LinSweep(a, 0, 3, outer_shape)
    sweep_b = LinSweep(b, 5, 7, outer_shape)
    sweep_c = LinSweep(c, 8, 12, inner_shape)

    datasets, _, _ = dond(
        TogetherSweep(sweep_a, sweep_b),
        sweep_c,
        dummyinstrument.A.dummy_parameter_with_setpoints,
        dummyinstrument.B.dummy_parameter_with_setpoints,
        dataset_dependencies={
            "ds1": (a, c, dummyinstrument.A.dummy_parameter_with_setpoints),
            "ds2": (b, c, dummyinstrument.B.dummy_parameter_with_setpoints),
        },
        do_plot=False,
    )
    assert isinstance(datasets, tuple)
    assert set(datasets[0].description.interdeps.names) == {
        "a",
        "c",
        "dummyinstrument_ChanA_dummy_sp_axis",
        "dummyinstrument_ChanA_dummy_parameter_with_setpoints",
    }
    assert datasets[0].description.shapes is not None
    assert len(datasets[0].description.shapes) == 1
    assert datasets[0].description.shapes[
        "dummyinstrument_ChanA_dummy_parameter_with_setpoints"
    ] == (outer_shape, inner_shape, n_points_a)

    assert set(datasets[1].description.interdeps.names) == {
        "b",
        "c",
        "dummyinstrument_ChanB_dummy_sp_axis",
        "dummyinstrument_ChanB_dummy_parameter_with_setpoints",
    }
    assert datasets[1].description.shapes is not None
    assert len(datasets[1].description.shapes) == 1
    assert datasets[1].description.shapes[
        "dummyinstrument_ChanB_dummy_parameter_with_setpoints"
    ] == (outer_shape, inner_shape, n_points_b)


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_together_sweep_parameter_with_setpoints_explicit_mapping_and_callable(
    dummyinstrument,
) -> None:
    outer_shape = 10
    inner_shape = 15

    n_points_a = 20
    dummyinstrument.A.dummy_n_points(n_points_a)
    n_points_b = 25
    dummyinstrument.B.dummy_n_points(n_points_b)

    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    sweep_a = LinSweep(a, 0, 3, outer_shape)
    sweep_b = LinSweep(b, 5, 7, outer_shape)
    sweep_c = LinSweep(c, 8, 12, inner_shape)

    datasets, _, _ = dond(
        TogetherSweep(sweep_a, sweep_b),
        sweep_c,
        lambda: print("this is a sideffect"),
        dummyinstrument.A.dummy_parameter_with_setpoints,
        dummyinstrument.B.dummy_parameter_with_setpoints,
        dataset_dependencies={
            "ds1": (a, c, dummyinstrument.A.dummy_parameter_with_setpoints),
            "ds2": (b, c, dummyinstrument.B.dummy_parameter_with_setpoints),
        },
        do_plot=False,
    )
    assert isinstance(datasets, tuple)
    assert set(datasets[0].description.interdeps.names) == {
        "a",
        "c",
        "dummyinstrument_ChanA_dummy_sp_axis",
        "dummyinstrument_ChanA_dummy_parameter_with_setpoints",
    }
    assert datasets[0].description.shapes is not None
    assert len(datasets[0].description.shapes) == 1
    assert datasets[0].description.shapes[
        "dummyinstrument_ChanA_dummy_parameter_with_setpoints"
    ] == (outer_shape, inner_shape, n_points_a)

    assert set(datasets[1].description.interdeps.names) == {
        "b",
        "c",
        "dummyinstrument_ChanB_dummy_sp_axis",
        "dummyinstrument_ChanB_dummy_parameter_with_setpoints",
    }
    assert datasets[1].description.shapes is not None
    assert len(datasets[1].description.shapes) == 1
    assert datasets[1].description.shapes[
        "dummyinstrument_ChanB_dummy_parameter_with_setpoints"
    ] == (outer_shape, inner_shape, n_points_b)


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_sweeper_combinations(_param_set, _param_set_2, _param) -> None:
    outer_shape = 10
    inner_shape = 15

    a = ManualParameter("a", initial_value=0)
    b = ManualParameter("b", initial_value=0)
    c = ManualParameter("c", initial_value=0)
    d = ManualParameter("d", initial_value=0)
    sweep_a = LinSweep(a, 0, 3, outer_shape)
    sweep_b = LinSweep(b, 5, 7, outer_shape)
    sweep_c = LinSweep(c, 8, 12, outer_shape)
    sweep_d = LinSweep(d, 13, 16, inner_shape)

    multi_sweep = TogetherSweep(sweep_a, sweep_b, sweep_c)

    sweeper = _Sweeper([multi_sweep, sweep_d], [])

    sweep_groups = sweeper.sweep_groupes

    expected_sweeper_groups = (
        (a, d),
        (b, d),
        (c, d),
        (a, b, d),
        (a, c, d),
        (b, c, d),
        (a, b, c, d),
    )

    assert len(expected_sweeper_groups) == len(sweep_groups)
    for g in expected_sweeper_groups:
        assert g in sweep_groups


@pytest.mark.usefixtures("plot_close", "experiment")
def test_sweep_int_vs_float() -> None:
    float_param = ManualParameter("float_param", initial_value=0.0)
    int_param = ManualParameter("int_param", vals=Ints(0, 100))

    dataset, _, _ = dond(ArraySweep(int_param, [1, 2, 3]), float_param)
    assert isinstance(dataset, DataSetProtocol)
    assert set(dataset.description.interdeps.names) == {"int_param", "float_param"}
    assert dataset.cache.data()["float_param"]["int_param"].dtype.kind == "i"


@pytest.mark.usefixtures("plot_close", "experiment")
def test_error_no_measured_parameters() -> None:
    float_param = ManualParameter("float_param", initial_value=0.0)
    int_param = ManualParameter("int_param", vals=Ints(0, 100))

    with pytest.raises(ValueError, match="No parameters to measure supplied"):
        dond(ArraySweep(int_param, [1, 2, 3]), ArraySweep(float_param, [1.0, 2.0, 3.0]))


@pytest.mark.usefixtures("plot_close", "experiment")
def test_error_measured_grouped_and_not_grouped() -> None:
    param_1 = ManualParameter("param_1", initial_value=0.0)
    param_2 = ManualParameter("param_2", initial_value=0.0)
    param_3 = ManualParameter("param_3", initial_value=0.0)

    with pytest.raises(
        ValueError, match="Got both grouped and non grouped parameters to measure in"
    ):
        dond(LinSweep(param_1, 0, 10, 10), param_2, [param_3])


@pytest.mark.usefixtures("plot_close", "experiment")
def test_post_action(mocker) -> None:
    param_1 = ManualParameter("param_1", initial_value=0.0)
    param_2 = ManualParameter("param_2", initial_value=0.0)

    post_actions = (mocker.MagicMock(),)
    dond(LinSweep(param_1, 0, 10, 10, post_actions=post_actions), param_2)

    post_actions[0].assert_called_with()


@pytest.mark.usefixtures("plot_close", "experiment")
def test_extra_log_info(caplog: LogCaptureFixture) -> None:
    param_1 = ManualParameter("param_1", initial_value=0.0)
    param_2 = ManualParameter("param_2", initial_value=0.0)

    log_message = "FOOBAR"
    with caplog.at_level(level=logging.INFO):
        dond(LinSweep(param_1, 0, 10, 10), param_2, log_info=log_message)

    assert log_message in caplog.text


@pytest.mark.usefixtures("plot_close", "experiment")
def test_default_log_info(caplog: LogCaptureFixture) -> None:
    param_1 = ManualParameter("param_1", initial_value=0.0)
    param_2 = ManualParameter("param_2", initial_value=0.0)

    with caplog.at_level(level=logging.INFO):
        dond(LinSweep(param_1, 0, 10, 10), param_2)

    assert "Using 'qcodes.dataset.dond'" in caplog.text


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_get_after_set(_param_set, _param_set_2, _param) -> None:
    n_points = 10

    a = TrackingParameter("a", initial_value=0)
    b = TrackingParameter("b", initial_value=0)

    a.reset_count()
    b.reset_count()

    assert a.get_count == 0
    assert a.set_count == 0
    assert b.get_count == 0
    assert b.set_count == 0

    dond(LinSweep(a, 0, 10, n_points, get_after_set=True), b)

    assert a.get_count == n_points
    assert a.set_count == n_points
    assert b.get_count == n_points
    assert b.set_count == 0


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_no_get_after_set(_param_set, _param_set_2, _param) -> None:
    n_points = 10

    a = TrackingParameter("a", initial_value=0)
    b = TrackingParameter("b", initial_value=0)

    a.reset_count()
    b.reset_count()

    assert a.get_count == 0
    assert a.set_count == 0
    assert b.get_count == 0
    assert b.set_count == 0

    dond(LinSweep(a, 0, 10, n_points, get_after_set=False), b)

    assert a.get_count == 0
    assert a.set_count == n_points
    assert b.get_count == n_points
    assert b.set_count == 0


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_get_after_set_stores_get_value(_param_set, _param_set_2, _param) -> None:
    n_points = 11

    a = GetReturnsCountParameter("a", initial_value=0)
    b = TrackingParameter("b", initial_value=0)

    a.reset_count()
    b.reset_count()

    assert a.get_count == 0
    assert a.set_count == 0
    assert b.get_count == 0
    assert b.set_count == 0

    ds, _, _ = dond(LinSweep(a, -10, -20, n_points, get_after_set=True), b)
    assert isinstance(ds, DataSetProtocol)
    # since we are using the GetReturnsCountParameter the sweep should be count e.g. 0, 1, ... 11
    # not the set parameters -10, .. - 20
    np.testing.assert_array_equal(
        ds.get_parameter_data()["b"]["a"], np.linspace(1, 11, n_points)
    )
    assert a.get_count == n_points
    assert a.set_count == n_points
    assert b.get_count == n_points
    assert b.set_count == 0


@pytest.mark.usefixtures("plot_close", "experiment")
def test_dond_return_type(_param_set, _param) -> None:
    n_points = 11

    # test that with squeeze=False we get MultiAxesTupleListWithDataSet as the return type
    dss, axs, cbs = dond(
        LinSweep(_param_set, -10, -20, n_points), _param, squeeze=False
    )

    assert isinstance(dss, tuple)
    assert_type(dss, tuple[DataSetProtocol, ...])
    assert len(dss) == 1
    assert isinstance(dss[0], DataSetProtocol)

    assert isinstance(axs, tuple)
    assert_type(
        axs,
        tuple[tuple["matplotlib.axes.Axes | None", ...], ...],
    )
    assert len(axs) == 1
    assert len(axs[0]) == 1
    assert axs[0][0] is None

    assert isinstance(cbs, tuple)
    assert_type(
        cbs,
        tuple[tuple["matplotlib.colorbar.Colorbar | None", ...], ...],
    )
    assert len(cbs) == 1
    assert len(cbs[0]) == 1
    assert cbs[0][0] is None
