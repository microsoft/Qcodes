"""
Bench mark suite intended to run with Airspeed-velocity:
http://asv.readthedocs.io/en/latest/

To run the benchmark, type:
asv run --python=same --quick --show-stderr --dry-run

in the command prompt. Make sure the environment you are running this in has the current Master installed as a package

"""
import numpy as np

from qcodes import ParamSpec, new_data_set, new_experiment


class TimeSuiteAddResults:
    """
    Make a moderately large data set and investigate insertion time. We make a single large results list and insert
    in one SQL command. Notice the plural in "Results"
    """

    def __init__(self):
        self._data_set = None
        self._insertion_size = 2000

    def setup(self):
        new_experiment("profile", "profile")
        self._data_set = new_data_set("stress_test_simple")

        t1 = ParamSpec('t', 'numeric', label='time', unit='s')
        x = ParamSpec('x', 'numeric', label='voltage', unit='v', depends_on=[t1])

        self._data_set.add_parameter(t1)
        self._data_set.add_parameter(x)

    def time_range(self):
        t_values = np.linspace(-1, 1, self._insertion_size)

        for _ in range(1000):
            results = [{"t": t, "x": 2 * t ** 2 + 1} for t in t_values]
            self._data_set.add_results(results)


class TimeSuiteAddResult:
    """
    Make a moderately large data set and investigate insertion time. We make a lot of single results and insert one by
    one. Notice the singular in "Result"
    """

    def __init__(self):
        self._data_set = None
        self._insertion_size = 200

    def setup(self):
        new_experiment("profile", "profile")
        self._data_set = new_data_set("stress_test_simple")

        t1 = ParamSpec('t', 'numeric', label='time', unit='s')
        x = ParamSpec('x', 'numeric', label='voltage', unit='v', depends_on=[t1])

        self._data_set.add_parameter(t1)
        self._data_set.add_parameter(x)

    def time_range(self):
        t_values = np.linspace(-1, 1, self._insertion_size)

        for _ in range(10):  # Adding results one by one is much more time consuming.
            results = [{"t": t, "x": 2 * t ** 2 + 1} for t in t_values]
            for result in results:
                self._data_set.add_result(result)


class TimeSuiteAddArrayResults:
    """
    Make a moderately large data set and investigate insertion time. We make a single large results list and insert
    in one SQL command. Notice the plural in "Results". The dependent parameters shall be arrays
    """

    def __init__(self):
        self._data_set = None
        self._insertion_size = 2000

    def setup(self):
        new_experiment("profile", "profile")
        self._data_set = new_data_set("stress_test_simple")

        t1 = ParamSpec('t', 'numeric', label='time', unit='s')
        x = ParamSpec('x', 'array', label='voltage', unit='v', depends_on=[t1])

        self._data_set.add_parameter(t1)
        self._data_set.add_parameter(x)

    def time_range(self):
        t_values = np.linspace(-1, 1, self._insertion_size)

        for _ in range(10):
            results = [{"t": t, "x": np.array([2 * t**2 + 1, t**3 - 1])} for t in t_values]
            self._data_set.add_results(results)
