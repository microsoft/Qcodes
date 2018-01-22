import cProfile
import matplotlib.pyplot as plt

import qcodes as qc
from qcodes import ParamSpec, new_data_set, new_experiment, load_data
import numpy as np

new_experiment("profile", "profile")


def add_results_stresstest(sz):

    data_set = new_data_set("stress_test_simple")
    t1 = ParamSpec('t', 'numeric', label='time', unit='s')
    x = ParamSpec('x', 'numeric', label='voltage', unit='v', depends_on=[t1])

    data_set.add_parameter(t1)
    data_set.add_parameter(x)

    t_values = np.linspace(-1, 1, sz)

    for _ in range(1000):
        results = [{"t": t, "x": 2 * t**2 + 1} for t in t_values]
        data_set.add_results(results)


def retrieve_results_stresstest():
    exp = qc.load_last_experiment()
    data = exp.data_set(1)
    x = data.get_values("x")


def time_performance():

    sample_sizes = [10, 50, 100, 150, 200, 250, 300, 350, 400, 500, 1000, 1200, 1400, 1600, 1800, 2000]
    add_total_times = []
    retrieve_total_times = []

    for sample_size in sample_sizes:
        profiler_add = cProfile.Profile()
        profiler_add.runcall(add_results_stresstest, sz=sample_size)

        profile_stats = profiler_add.getstats()
        add_times = [r.totaltime / 1000 * 1000 for r in profile_stats if "sqlite_base.py" in str(r.code)]
        add_total_times.append(sum(add_times))

        profiler_retrieve = cProfile.Profile()
        profiler_retrieve.runcall(retrieve_results_stresstest)

        profile_stats = profiler_retrieve.getstats()
        retrieve_times = [r.totaltime / 1000 * 1000 for r in profile_stats if "sqlite_base.py" in str(r.code)]
        retrieve_total_times.append(sum(retrieve_times))

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(sample_sizes, add_total_times)
    ax1.set_xlabel("sample size")
    ax1.set_ylabel("average 'add_results' time [ms]")

    ax2.plot(sample_sizes, retrieve_total_times)
    ax2.set_xlabel("sample size")
    ax2.set_ylabel("average 'get_values' time [ms]")

    plt.show()

if __name__ == "__main__":
    time_performance()

