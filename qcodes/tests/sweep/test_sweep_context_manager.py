"""
A context manager subclassed from the original has been designed to work seamlessly with sweep objects. This is tested
here.
"""
import numpy as np
import itertools

from qcodes.instrument.parameter import ManualParameter
from qcodes.sweep import sweep, nest, chain, SweepMeasurement, setter
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
    meas.register_sweep(sweep_object)

    with meas.run() as datasaver:
        for data in sweep_object:
            datasaver.add_result(*data.items())

    data_set = datasaver._dataset
    assert data_set.paramspecs["x"].depends_on == ""
    assert data_set.paramspecs["m"].depends_on == "x"

    expected_x = [[xi] for xi in sweep_values]
    assert data_set.get_data('x') == expected_x


def test_inferred():

    x = ManualParameter("x", unit="V")

    @setter([("xmv", "mV")], inferred_parameters=[("x", "V")])
    def xsetter(milivolt_value):
        volt_value = milivolt_value / 1000.0  # From mV to V
        x.set(volt_value)
        return volt_value

    m = ManualParameter("m")
    m.get = lambda: np.sin(x())

    sweep_values = np.linspace(-1000, 1000, 100)  # We sweep in mV

    sweep_object = nest(sweep(xsetter, sweep_values), m)

    experiment = new_experiment("sweep_measure", sample_name="sine")
    station = Station()
    meas = SweepMeasurement(exp=experiment, station=station)
    meas.register_sweep(sweep_object)

    with meas.run() as datasaver:
        for data in sweep_object:
            datasaver.add_result(*data.items())

    data_set = datasaver._dataset
    assert data_set.paramspecs["x"].depends_on == ""
    assert data_set.paramspecs["x"].inferred_from == "xmv"
    assert data_set.paramspecs["xmv"].depends_on == ""
    assert data_set.paramspecs["m"].depends_on == "x"

    expected_xmv = [[xi] for xi in sweep_values]
    expected_x = [[xi / 1000] for xi in sweep_values]

    assert data_set.get_data('xmv') == expected_xmv
    assert data_set.get_data('x') == expected_x


def test_nest():
    n_sample_points = 100
    x = ManualParameter("x")
    sweep_values_x = np.linspace(-1, 1, n_sample_points)

    y = ManualParameter("y")
    sweep_values_y = np.linspace(-1, 1, n_sample_points)

    m = ManualParameter("m")
    m.get = lambda: np.sin(x())

    n = ManualParameter("n")
    n.get = lambda: np.cos(x()) + 2 * np.sin(y())

    sweep_object = sweep(x, sweep_values_x)(
        m,
        sweep(y, sweep_values_y)(
            n
        )
    )

    experiment = new_experiment("sweep_measure", sample_name="sine")
    station = Station()
    meas = SweepMeasurement(exp=experiment, station=station)
    meas.register_sweep(sweep_object)

    with meas.run() as datasaver:
        for data in sweep_object:
            datasaver.add_result(*data.items())

    data_set = datasaver._dataset
    assert data_set.paramspecs["x"].depends_on == ""
    assert data_set.paramspecs["y"].depends_on == ""
    assert data_set.paramspecs["m"].depends_on == "x"
    assert data_set.paramspecs["n"].depends_on == "x, y"

    data_x = data_set.get_data('x')
    data_y = data_set.get_data('y')

    assert data_x[::n_sample_points + 1] == [[xi] for xi in sweep_values_x]
    assert data_y[::n_sample_points + 1] == [[None] for _ in sweep_values_x]

    coordinate_layout = itertools.product(sweep_values_x, sweep_values_y)
    expected_x, expected_y = zip(*coordinate_layout)
    assert [ix for c, ix in enumerate(data_x)
            if c % (n_sample_points + 1)] == [[xi] for xi in expected_x]

    assert [iy for c, iy in enumerate(data_y)
            if c % (n_sample_points + 1)] == [[yi] for yi in expected_y]




