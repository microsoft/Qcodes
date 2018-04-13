"""
Bench mark suite intended to run with Airspeed-velocity:
http://asv.readthedocs.io/en/latest/

To run the benchmark, type:
asv run

in the command prompt.

It is also possible to run this file as a standalone script without the need
to have airspeed-velocity installed. Running the main function will
automatically generate a report in reStructuredText format. The report will
show benchmark of the various defined in this file.
"""
import numpy as np
import timeit
import matplotlib.pyplot as plt
import inspect

from qcodes import ParamSpec, new_data_set, new_experiment
from qcodes.dataset.measurements import Measurement
from qcodes.instrument.parameter import ManualParameter


class TimeSuite:
    """
    Time suite benchmark prototype
    """
    params = None
    repeats = None

    def __init__(self, insertion_size: int) ->None:
        raise NotImplementedError

    def setup(self, insertion_size: int) ->None:
        raise NotImplementedError

    def time_range(self, insertion_size: int) ->None:
        raise NotImplementedError


class TimeSuiteAddResults(TimeSuite):
    params = [200, 500, 1000, 1500, 2000, 2500, 3000]
    repeats = 1000

    def __init__(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        super().__init__(insertion_size)
        self._data_set = None

    def setup(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        new_experiment("profile", "profile")
        self._data_set = new_data_set("stress_test_simple")

        t1 = ParamSpec('t', 'numeric', label='time', unit='s')
        x = ParamSpec(
            'x', 'numeric', label='voltage', unit='v', depends_on=[t1]
        )

        self._data_set.add_parameter(t1)
        self._data_set.add_parameter(x)

    def time_range(self, insertion_size: int) ->None:
        """
        We test the insertion time as a function of the number of results we
        generate. Add all results in one sql command using the "add_results"
        method (notice the plural "s")

        Args:
            insertion_size (int): The number of results to generate
        """
        t_values = np.linspace(-1, 1, insertion_size)
        results = [{"t": t, "x": 2 * t ** 2 + 1} for t in t_values]
        self._data_set.add_results(results)


class TimeSuiteAddResult(TimeSuite):
    """
    Make a moderately large data set and investigate insertion time. We make a
    lot of single results and insert one by
    one. Notice the singular in "Result".
    """
    params = [20, 50, 100, 150, 200, 250, 300]
    repeats = 10

    def __init__(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        super().__init__(insertion_size)
        self._data_set = None

    def setup(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        new_experiment("profile", "profile")
        self._data_set = new_data_set("stress_test_simple")

        t1 = ParamSpec('t', 'numeric', label='time', unit='s')
        x = ParamSpec(
            'x', 'numeric', label='voltage', unit='v', depends_on=[t1]
        )

        self._data_set.add_parameter(t1)
        self._data_set.add_parameter(x)

    def time_range(self, insertion_size: int) ->None:
        """
        We test the insertion time as a function of the number of results we
        generate. Then, add the results in one by one on a loop by calling
        "add_result". Contrast this with the plot "TimeSuiteAddResults"; we
        see that this method is ~200 times slower!

        Args:
            insertion_size (int): The number of results to generate
        """
        t_values = np.linspace(-1, 1, insertion_size)
        results = [{"t": t, "x": 2 * t ** 2 + 1} for t in t_values]

        for result in results:
            self._data_set.add_result(result)


class TimeSuiteAddResultContext(TimeSuite):
    params = [20, 50, 100, 150, 200, 250, 300]
    repeats = 100

    def __init__(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        super().__init__(insertion_size)
        self._meas = None
        self._x = None
        self._m = None

    def setup(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        new_experiment("profile", "profile")
        self._meas = Measurement()
        x = ManualParameter("x")
        m = ManualParameter("m")

        self._meas.register_parameter(x)
        self._meas.register_parameter(m, setpoints=(x,))

        self._x = x
        self._m = m

    def time_range(self, insertion_size: int) ->None:
        """
        Use the context manager to add results in a data set. Compare this
        result with the "TimeSuiteAddResult" and "TimeSuiteAddResults". We see
        that although it is not as slow as the former, it is still much slower
        then the latter. TODO: We should find out why this is so much slower.

        Args:
            insertion_size (int): The number of results to generate
        """
        with self._meas.run() as datasaver:
            for ix, im in zip(range(insertion_size), range(insertion_size)):
                datasaver.add_result((self._x, ix), (self._m, im))


class TimeSuiteAddArrayResults(TimeSuite):
    """
    Make a moderately large data set and investigate insertion time. We make a
    single large results list and insert in one SQL command. Notice the plural
    in "Results". The dependent parameters shall be arrays
    """
    params = [200, 500, 1000, 1500, 2000, 2500, 3000]
    repeats = 100

    def __init__(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        super().__init__(insertion_size)
        self._data_set = None

    def setup(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        new_experiment("profile", "profile")
        self._data_set = new_data_set("stress_test_simple")

        t1 = ParamSpec('t', 'numeric', label='time', unit='s')
        x = ParamSpec('x', 'array', label='voltage', unit='v', depends_on=[t1])

        self._data_set.add_parameter(t1)
        self._data_set.add_parameter(x)

    def time_range(self, insertion_size: int) ->None:
        """
        Insert arrayed valued values. Each result contains a 1x2 array. Again
        we see that this is much slower then inserting single valued results.

        Args:
            insertion_size (int): The number of results to generate
        """
        t_values = np.linspace(-1, 1, insertion_size)
        results = [{"t": t, "x": np.array([2 * t**2 + 1, t**3 - 1])} for t in
                   t_values]
        self._data_set.add_results(results)


class TimeSuiteAddArrayResultsII(TimeSuite):
    """
    Make a moderately large data set and investigate insertion time. We make a
    single large results list and insert in one SQL command. Notice the plural
    in "Results". The dependent parameters shall be arrays
    """
    params = [200, 500, 1000, 1500, 2000, 2500, 3000]
    repeats = 100

    def __init__(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        super().__init__(insertion_size)
        self._data_set = None

    def setup(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        new_experiment("profile", "profile")
        self._data_set = new_data_set("stress_test_simple")

        t1 = ParamSpec('t', 'numeric', label='time', unit='s')
        x = ParamSpec('x', 'array', label='voltage', unit='v', depends_on=[t1])

        self._data_set.add_parameter(t1)
        self._data_set.add_parameter(x)

    def time_range(self, insertion_size: int) ->None:
        """
        Insert arrayed valued values. The dimensionality of the array increases
        along the x-axis.

        Args:
            insertion_size (int): The number of results to generate
        """
        t_values = np.linspace(-1, 1, 1000)
        results = [{"t": t, "x": np.random.uniform(0, 1, (1, insertion_size))}
                   for t in t_values]
        self._data_set.add_results(results)


class TimeSuiteParamCount(TimeSuite):
    params = [2, 5, 10, 20, 50, 70, 100]
    repeats = 10

    def __init__(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        super().__init__(insertion_size)
        self._data_set = None
        self._results = None

    def setup(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        new_experiment("profile", "profile")
        self._data_set = new_data_set("stress_test_simple")

        t = ParamSpec('t', 'numeric', label='time', unit='s')
        self._data_set.add_parameter(t)

        for x in [
            ParamSpec(
                "x_{n}".format(n=str(n)), "numeric",
                label="x_{n}".format(n=str(n)), unit="V", depends_on=[t]
            )
            for n in range(insertion_size)
        ]:
            self._data_set.add_parameter(x)

    def time_range(self, insertion_size: int) ->None:
        """
        Investigate the insertion time as a function of the number of
        parameters

        Args:
            insertion_size (int): : The number of results to generate
        """
        results = []
        xdict = {"x_{n}".format(n=str(n)): 0 for n in range(insertion_size)}

        for t in np.linspace(-1, 1, 1000):
            r = {"t": t}
            r.update(xdict)
            results.append(r)

        self._data_set.add_results(results)


class TimeSuiteAddArrayResultsContext(TimeSuite):
    params = [200, 500, 1000, 1500, 2000, 2500, 3000]
    repeats = 100

    def __init__(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        super().__init__(insertion_size)
        self._meas = None
        self._x = None
        self._m = None

    def setup(self, insertion_size: int) ->None:
        """
        Args:
             insertion_size (int): The number of results to generate. Although
             this argument is not used in the init, the airspeed velocity
             frame work expects an argument.
        """
        new_experiment("profile", "profile")
        self._meas = Measurement()
        x = ManualParameter("x")
        m = ManualParameter("m")

        self._meas.register_parameter(x)
        self._meas.register_parameter(m, setpoints=(x,))

        self._x = x
        self._m = m

    def time_range(self, insertion_size: int) ->None:
        """
        Add array valued results with the context manager.

        Args:
            insertion_size (int): The number of results to generate
        """
        self._x(0)
        self._m.get = lambda: np.arange(insertion_size)

        with self._meas.run() as datasaver:
            datasaver.add_result((self._x, self._x()), (self._m, self._m()))


def run_suite(suite_cls: TimeSuite) ->str:
    """
    Run a time suite class. For each parameter in suite_cls.params, run the
    setup method and use the timeit method to get an estimate of how long the
    execution time is. To get a better estimate we tell the timeit module to
    run the test a number of times, given by suite_cls.repeats. We then plot
    the average running time as a function of the parameter and save the result
    to a image file in png format.

    Args:
        suite_cls (type): The time suite class to run

    Returns:
        result_plot_file_name (str): The file name the results of the
        benchmark are saved in
    """
    params = suite_cls.params
    repeats = suite_cls.repeats
    suite_name = suite_cls.__name__
    t_per_param = []

    print("running suite {}".format(suite_name))

    for p in params:

        setup_str = ";".join([
            f"from __main__ import {suite_name} as suite_cls",
            f"suite = suite_cls({p})",
            f"suite.setup({p})"
        ])

        t = timeit.timeit(
            f"suite.time_range({p})",
            setup=setup_str,
            number=repeats
        )

        t_at_param = t / repeats
        t_per_param.append(t_at_param)

    fig, ax = plt.subplots()
    ax.plot(params, t_per_param, ".-")
    ax.set_xlabel("insertion_size")
    ax.set_ylabel("timeit [s]")
    ax.set_title("{}".format(suite_name))

    result_plot_file_name = "{}.png".format(suite_name)
    plt.savefig(result_plot_file_name)
    print("done")
    return result_plot_file_name


def make_report():
    """
    Run the benchmark suites and make a report in reStructuredText format.
    """
    report = "Benchmark results \n=================\n"

    suites = [
        TimeSuiteAddResults, TimeSuiteAddResult, TimeSuiteAddResultContext,
        TimeSuiteAddArrayResults, TimeSuiteAddArrayResultsII,
        TimeSuiteAddArrayResultsContext, TimeSuiteParamCount
    ]

    for suite in suites:
        result_plot_file_name = run_suite(suite)
        code = inspect.getsource(suite.time_range)

        report += "\n\t".join([
            f".. image:: {result_plot_file_name}",
            ":width: 1000px",
            ":align: center",
            ":height: 800px",
            "\n.. code-block:: python",
            f"\n{code}"
        ])

    with open("report.rst", "w") as fh:
        fh.write(report)

if __name__ == "__main__":
    make_report()
