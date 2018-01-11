"""
A context manager subclassed from the original has been designed to work seamlessly with sweep objects. This is tested
here.
"""
import numpy as np
import itertools

from qcodes.instrument.parameter import ManualParameter
from qcodes.sweep import sweep, nest, chain, SweepMeasurement
from qcodes import new_experiment, Station


def test_simple():
    x = ManualParameter("x")
    sweep_values = np.linspace(-1, 1, 100)

    m = ManualParameter("m")
    m.get = lambda: np.sin(x())

    sweep_object = nest(sweep(x, sweep_values), m)

    experiment = new_experiment("sweep_measure", sample_name="sine")
    station = Station()
    meas = SweepMeasurement(exp=experiment, station=station)

    with meas.run() as datasaver:
        for data in sweep_object:
            datasaver.add_result(data)

    data_set = datasaver._dataset
    assert data_set.paramspecs["x"].depends_on == ""
    assert data_set.paramspecs["m"].depends_on == "x"

    expected_x = [[xi] for xi in sweep_values]
    assert data_set.get_data('x') == expected_x


def test_nest():
    x = ManualParameter("x")
    sweep_values_x = np.linspace(-1, 1, 10)  # We have only 10 data points along each axis because of the bug described
    # in https://github.com/QCoDeS/Qcodes/issues/942

    y = ManualParameter("y")
    sweep_values_y = np.linspace(-1, 1, 10)

    m = ManualParameter("m")
    m.get = lambda: np.sin(x())

    n = ManualParameter("n")
    n.get = lambda: np.cos(x()) + 2 * np.sin(y())

    sweep_object = nest(sweep(x, sweep_values_x), chain(m, nest(sweep(y, sweep_values_y), n)))
    # x * (m + y * n)

    experiment = new_experiment("sweep_measure", sample_name="sine")
    station = Station()
    meas = SweepMeasurement(exp=experiment, station=station)

    with meas.run() as datasaver:
        for data in sweep_object:
            datasaver.add_result(data)

    data_set = datasaver._dataset
    assert data_set.paramspecs["x"].depends_on == ""
    assert data_set.paramspecs["y"].depends_on == ""
    assert data_set.paramspecs["m"].depends_on == "x"
    assert data_set.paramspecs["n"].depends_on == "y, x"

    data_x = data_set.get_data('x')
    data_y = data_set.get_data('y')

    assert data_x[::11] == [[xi] for xi in sweep_values_x]
    assert data_y[::11] == [[None] for _ in sweep_values_x]

    coordinate_layout = itertools.product(sweep_values_x, sweep_values_y)
    expected_x, expected_y = zip(*coordinate_layout)
    assert [ix for c, ix in enumerate(data_x) if c % 11] == [[xi] for xi in expected_x]
    assert [iy for c, iy in enumerate(data_y) if c % 11] == [[yi] for yi in expected_y]




