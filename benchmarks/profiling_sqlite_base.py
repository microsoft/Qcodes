import cProfile
import pstats
import timeit
import time
from statistics import mean

import qcodes as qc

from qcodes import ParamSpec, new_data_set, new_experiment
from qcodes.dataset.measurements import Measurement
from qcodes.instrument.parameter import ManualParameter

import numpy as np
import timeit
import matplotlib.pyplot as plt
import inspect


def benchmark_add_results_vs_MAX_VARIABLE_NUMBER():
    plt.figure()
    xr, yr = [], []
    filename = 'benchmark_add_results_vs_MAX_VARIABLE_NUMBER.png'

    mvn = qc.SQLiteSettings.limits['MAX_VARIABLE_NUMBER']
    for i in range(2, mvn, mvn//50):
        ts = []
        for j in range(3):
            qc.SQLiteSettings.limits['MAX_VARIABLE_NUMBER'] = i
            new_experiment("profile", "profile")
            data_set = new_data_set("stress_test_simple")

            t1 = ParamSpec('t', 'numeric', label='time', unit='s')
            x = ParamSpec('x', 'numeric',
                          label='voltage', unit='v', depends_on=[t1])

            data_set.add_parameter(t1)
            data_set.add_parameter(x)
            insertion_size = 400 * 600
            t_values = np.linspace(-1, 1, insertion_size)
            results = [{"t": t, "x": 2 * t ** 2 + 1} for t in t_values]

            t1r = time.time()
            data_set.add_results(results)
            t = time.time() - t1r
            ts.append(t)
        xr.append(i)
        yr.append(mean(ts))

    plt.plot(xr, yr)
    plt.ylabel('execution time of data_set.add_results(result)')
    plt.xlabel('MAX_VARIABLE_NUMBER')
    plt.savefig(filename)
    return filename


if __name__ == '__main__':
    benchmark_add_results_vs_MAX_VARIABLE_NUMBER()
