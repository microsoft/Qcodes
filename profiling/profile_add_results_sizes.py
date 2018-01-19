"""
In this script, we profile the time performance of the "add_results" call as a function of the number of samples we
insert.
"""

from collections import defaultdict
import cProfile
import matplotlib.pyplot as plt

from qcodes import ParamSpec, new_data_set, new_experiment
import numpy as np

new_experiment("profile", "profile")


def stress_test_simple(sz):

    data_set = new_data_set("stress_test_simple")
    t1 = ParamSpec('t', 'real', label='time', unit='s')
    x = ParamSpec('x', 'real', label='voltage', unit='v', depends_on=[t1])

    data_set.add_parameter(t1)
    data_set.add_parameter(x)

    t_values = np.linspace(-1, 1, sz)

    results = [{"t": t, "x": 2 * t**2 + 1} for t in t_values]
    data_set.add_results(results)


def run_profiler(sz):
    profiler = cProfile.Profile()
    profiler.runcall(stress_test_simple, sz=sz)
    return profiler.getstats()


def time_performance():

    sample_sizes = [10, 50, 100, 150, 200, 250, 300, 350, 400]  # We cannot make the sample sizes bigger yet due to
    # bug https://github.com/QCoDeS/Qcodes/issues/942
    total_times = defaultdict(lambda: 0)

    repeats = 1000

    for _ in range(repeats):
        for sample_size in sample_sizes:
            profiler = cProfile.Profile()
            profiler.runcall(stress_test_simple, sz=sample_size)

            profile_stats = profiler.getstats()
            times = [r.totaltime/1000 for r in profile_stats if "sqlite_base.py" in str(r.code)]
            total_times[sample_size] += sum(times)

    tt = [v/repeats for v in total_times.values()]

    plt.plot(sample_sizes, tt)
    plt.xlabel("sample size")
    plt.ylabel("average 'add_results' time [ms]")
    plt.show()

if __name__ == "__main__":
    time_performance()