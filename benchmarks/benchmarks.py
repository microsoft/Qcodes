"""
Bench mark suite intended to run with Airspeed-velocity:
http://asv.readthedocs.io/en/latest/

To run the benchmark, type:
asv run

in the command prompt.

It is also possible to run this file as a standalone script without the need to have airspeed-velocity installed.
Running the main function will automatically generate a report in reStructuredText format. The report will show
benchmark of the various defined in this file.
"""
import numpy as np
import timeit
import matplotlib.pyplot as plt
import inspect

from qcodes import ParamSpec, new_data_set, new_experiment
from qcodes.dataset.measurements import Measurement
from qcodes.instrument.parameter import ManualParameter


class TimeSuiteAddResults:
    """
    Make a moderately large data set and investigate insertion time. We make a single large results list and insert
    in one SQL command. Notice the plural in "Results"
    """
    params = [200, 500, 1000, 1500, 2000]
    repeats = 1000

    def __init__(self, insertion_size):
        self._data_set = None

    def setup(self, insertion_size):
        new_experiment("profile", "profile")
        self._data_set = new_data_set("stress_test_simple")

        t1 = ParamSpec('t', 'numeric', label='time', unit='s')
        x = ParamSpec('x', 'numeric', label='voltage', unit='v', depends_on=[t1])

        self._data_set.add_parameter(t1)
        self._data_set.add_parameter(x)

    def time_range(self, insertion_size):
        t_values = np.linspace(-1, 1, insertion_size)
        results = [{"t": t, "x": 2 * t ** 2 + 1} for t in t_values]
        self._data_set.add_results(results)


class TimeSuiteAddResultContext:
    params = [200, 500, 1000, 1500, 2000]
    repeats = 10

    def __init__(self, insertion_size):
        self._meas = None
        self._x = None
        self._m = None

    def setup(self, insertion_size):
        new_experiment("profile", "profile")
        self._meas = Measurement()
        x = ManualParameter("x")
        m = ManualParameter("m")

        self._meas.register_parameter(x)
        self._meas.register_parameter(m, setpoints=(x,))

        self._x = x
        self._m = m

    def time_range(self, insertion_size):
        self._x(0)
        self._m.get = lambda: np.arange(insertion_size)

        with self._meas.run() as datasaver:
            datasaver.add_result((self._x, self._x()), (self._m, self._m()))


class TimeSuiteAddResult:
    """
    Make a moderately large data set and investigate insertion time. We make a lot of single results and insert one by
    one. Notice the singular in "Result"
    """
    params = [20, 50, 100, 150, 200]
    repeats = 10

    def __init__(self, insertion_size):
        self._data_set = None

    def setup(self, insertion_size):
        new_experiment("profile", "profile")
        self._data_set = new_data_set("stress_test_simple")

        t1 = ParamSpec('t', 'numeric', label='time', unit='s')
        x = ParamSpec('x', 'numeric', label='voltage', unit='v', depends_on=[t1])

        self._data_set.add_parameter(t1)
        self._data_set.add_parameter(x)

    def time_range(self, insertion_size):

        t_values = np.linspace(-1, 1, insertion_size)
        results = [{"t": t, "x": 2 * t ** 2 + 1} for t in t_values]

        for result in results:
            self._data_set.add_result(result)


class TimeSuiteAddArrayResults:
    """
    Make a moderately large data set and investigate insertion time. We make a single large results list and insert
    in one SQL command. Notice the plural in "Results". The dependent parameters shall be arrays
    """
    params = [200, 500, 1000, 1500, 2000]
    repeats = 10

    def __init__(self, insertion_size):
        self._data_set = None

    def setup(self, insertion_size):
        new_experiment("profile", "profile")
        self._data_set = new_data_set("stress_test_simple")

        t1 = ParamSpec('t', 'numeric', label='time', unit='s')
        x = ParamSpec('x', 'array', label='voltage', unit='v', depends_on=[t1])

        self._data_set.add_parameter(t1)
        self._data_set.add_parameter(x)

    def time_range(self, insertion_size):
        t_values = np.linspace(-1, 1, insertion_size)
        results = [{"t": t, "x": np.array([2 * t**2 + 1, t**3 - 1])} for t in t_values]
        self._data_set.add_results(results)


def run_suite(suite_cls):

    params = suite_cls.params
    repeats = suite_cls.repeats
    suite_name = suite_cls.__name__
    t_per_param = []

    print("running suite {}".format(suite_name))

    for p in params:

        t = timeit.timeit(
            "suite.time_range(p)",
            setup="from __main__ import {name} as suite_cls; "
                  "p={p}; suite = suite_cls(p); suite.setup(p)".format(p=p, name=suite_name),
            number=repeats)

        t_at_param = t / repeats
        t_per_param.append(t_at_param)

    fig, ax = plt.subplots()
    ax.plot(params, t_per_param)
    ax.set_xlabel("insertion_size")
    ax.set_ylabel("timeit [s]")
    ax.set_title("{}".format(suite_name))

    result_plot_file_name = "{}.png".format(suite_name)
    plt.savefig(result_plot_file_name)
    print("done")
    return result_plot_file_name


def make_report():

    report = "Benchmark results \n=================\n"

    suites = [TimeSuiteAddResultContext, TimeSuiteAddResults, TimeSuiteAddResult, TimeSuiteAddArrayResults]

    for suite in suites:
        #result_plot_file_name = run_suite(suite)
        result_plot_file_name = "{}.png".format(suite.__name__)

        code = inspect.getsource(suite.time_range)

        report += f".. image:: {result_plot_file_name}\n\t:width: 1000px\n\t:align: center\n\t:height: 800px"
        report += f"\n\n.. code-block:: python\n\n{code}"

    with open("report.rst", "w") as fh:
        fh.write(report)

if __name__ == "__main__":
    make_report()
