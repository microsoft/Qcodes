from collections import OrderedDict

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from qcodes.instrument.parameter import combine
from qcodes.utils.helpers import full_class

from ..common import DumyPar


@pytest.fixture()
def parameters():
    parameters = [DumyPar(name) for name in ["X", "Y", "Z"]]
    yield parameters


def testCombine(parameters):
    multipar = combine(*parameters, name="combined")
    assert multipar.dimensionality == len(parameters)


def testSweepBadSetpoints(parameters):
    with pytest.raises(ValueError):
        combine(*parameters, name="fail").sweep(np.array([[1, 2]]))


def testSweep(parameters):
    setpoints = np.array([[1, 1, 1], [1, 1, 1]])

    sweep_values = combine(*parameters,
                           name="combined").sweep(setpoints)

    res = []
    for i in sweep_values:
        value = sweep_values.set(i)
        res.append([i, value])
    expected = [
            [0, [1, 1, 1]],
            [1, [1, 1, 1]]
            ]
    assert res == expected


def testSet(parameters, mocker):
    setpoints = np.array([[1, 1, 1], [1, 1, 1]])

    sweep_values = combine(*parameters,
                           name="combined").sweep(setpoints)

    mock_method = mocker.patch.object(sweep_values, 'set')
    for i in sweep_values:
        sweep_values.set(i)

    mock_method.assert_has_calls([
                mocker.call(0), mocker.call(1)
            ]
        )


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(npoints=hst.integers(1, 100),
       x_start_stop=hst.lists(hst.integers(), min_size=2, max_size=2).map(sorted),
       y_start_stop=hst.lists(hst.integers(), min_size=2, max_size=2).map(sorted),
       z_start_stop=hst.lists(hst.integers(), min_size=2, max_size=2).map(sorted))
def testAggregator(parameters, npoints, x_start_stop, y_start_stop, z_start_stop):

    x_set = np.linspace(x_start_stop[0], x_start_stop[1], npoints).reshape(npoints, 1)
    y_set = np.linspace(y_start_stop[0], y_start_stop[1], npoints).reshape(npoints, 1)
    z_set = np.linspace(z_start_stop[0], z_start_stop[1], npoints).reshape(npoints, 1)
    setpoints = np.hstack((x_set, y_set, z_set))
    expected_results = [linear(*set) for set in setpoints]
    sweep_values = combine(*parameters,
                           name="combined",
                           aggregator=linear).sweep(setpoints)

    results = []
    for i, value in enumerate(sweep_values):
            res = sweep_values.set(value)
            results.append(sweep_values._aggregate(*res))

    assert results == expected_results


def testMeta(parameters):
    name = "combined"
    label = "Linear Combination"
    unit = "a.u"
    aggregator = linear
    sweep_values = combine(*parameters,
                           name=name,
                           label=label,
                           unit=unit,
                           aggregator=aggregator
                           )
    snap = sweep_values.snapshot()
    out = OrderedDict()
    out['__class__'] = full_class(sweep_values)
    out["unit"] = unit
    out["label"] = label
    out["full_name"] = name
    out["aggregator"] = repr(linear)
    for param in sweep_values.parameters:
        out[param.full_name] = {}
    assert out == snap


def testMutable(parameters):
    setpoints = np.array([[1, 1, 1], [1, 1, 1]])

    sweep_values = combine(*parameters,
                           name="combined")
    a = sweep_values.sweep(setpoints)
    setpoints = np.array([[2, 1, 1], [1, 1, 1]])
    b = sweep_values.sweep(setpoints)
    assert a != b


def testArrays(parameters):
    x_vals = np.linspace(1, 1, 2)
    y_vals = np.linspace(1, 1, 2)
    z_vals = np.linspace(1, 1, 2)
    sweep_values = combine(*parameters,
                           name="combined").sweep(x_vals, y_vals, z_vals)
    res = []
    for i in sweep_values:
        value = sweep_values.set(i)
        res.append([i, value])

    expected = [
            [0, [1, 1, 1]],
            [1, [1, 1, 1]]
            ]
    assert res == expected


def testWrongLen(parameters):
    x_vals = np.linspace(1, 1, 2)
    y_vals = np.linspace(1, 1, 2)
    z_vals = np.linspace(1, 1, 3)
    with pytest.raises(ValueError):
        combine(*parameters,
                name="combined").sweep(x_vals, y_vals, z_vals)


def testInvalidName(parameters):
    x_vals = np.linspace(1, 1, 2)
    y_vals = np.linspace(1, 1, 2)
    z_vals = np.linspace(1, 1, 2)
    with pytest.raises(ValueError):
        combine(*parameters,
                name="combined with spaces").sweep(x_vals, y_vals, z_vals)


def testLen(parameters):
    x_vals = np.linspace(1, 1, 2)
    y_vals = np.linspace(1, 1, 2)
    z_vals = np.linspace(1, 0, 2)
    sweep_values = combine(*parameters,
                           name="combined").sweep(x_vals, y_vals, z_vals)
    assert len(x_vals) == len(sweep_values.setpoints)


def linear(x, y, z):
    return x+y+z
