import os
from shutil import copyfile
import numpy as np

from qcodes import ParamSpec, new_data_set, new_experiment, load_last_experiment, DataSet
from qcodes.dataset.experiment_container import Experiment


class TimeSuiteInsertion:
    """
    Make a moderately large data set and investigate insertion time.
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

    def teardown(self):
        if not os.path.exists("experiments_persistent.db"):
            copyfile("experiments.db", "experiments_persistent.db")

    def time_range(self):
        t_values = np.linspace(-1, 1, self._insertion_size)

        for _ in range(1000):
            results = [{"t": t, "x": 2 * t ** 2 + 1} for t in t_values]
            self._data_set.add_results(results)


class TimeSuiteExtraction:
    """
    Monitor how the data retrieval time evolves in time. Specifically, we do not want the retrieval time to increase
    unacceptably when the data set becomes really large
    """

    def time_range(self):
        exp = load_last_experiment()
        data = exp.data_set(1)
        x = data.get_values("x")
