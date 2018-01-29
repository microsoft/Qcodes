import numpy as np

from qcodes import ParamSpec, new_data_set, new_experiment


class TimeSuite:
    """
    Make a moderately large data set and investigate insertion time.
    """
    inseration_size = 2000

    def __init__(self):
        self._data_set = None

    def setup(self):
        new_experiment("profile", "profile")
        self._data_set = new_data_set("stress_test_simple")
        t1 = ParamSpec('t', 'numeric', label='time', unit='s')
        x = ParamSpec('x', 'numeric', label='voltage', unit='v', depends_on=[t1])

        self._data_set.add_parameter(t1)
        self._data_set.add_parameter(x)

    def time_range(self):
        t_values = np.linspace(-1, 1, 2000)

        for _ in range(1000):
            results = [{"t": t, "x": 2 * t ** 2 + 1} for t in t_values]
            self._data_set.add_results(results)
