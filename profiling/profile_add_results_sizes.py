"""
In this script, we profile the time performance of the "add_results" call as a function of the number of samples we
insert.
"""

import cProfile
from memory_profiler import profile as memory_profiler

from qcodes import ParamSpec, new_data_set, new_experiment
import numpy as np

from profiling.utilities import capture_stdout, parse_profile_result

new_experiment("profile", "profile")

@capture_stdout
@memory_profiler
def stress_test_simple(sz):

    data_set = new_data_set("stress_test_simple")
    t1 = ParamSpec('t', 'real', label='time', unit='s')
    x = ParamSpec('x', 'real', label='voltage', unit='v', depends_on=[t1])

    data_set.add_parameter(t1)
    data_set.add_parameter(x)

    t_values = np.linspace(-1, 1, sz)

    results = [{"t": t, "x": 2 * t**2 + 1} for t in t_values]
    data_set.add_results(results)


@capture_stdout
def run_profiler(sz):
    cProfile.run("stress_test_simple({sz})".format(sz=sz))


def time_performance():

    sample_sizes = [10, 50, 100, 150, 200, 250, 300, 350, 400]  # We cannot make the sample sizes bigger yet due to
    # bug https://github.com/QCoDeS/Qcodes/issues/942
    total_times = []

    for sample_size in sample_sizes:
        profile_result = run_profiler(sample_size)
        parsed_profile_result = parse_profile_result(profile_result)

        line_count = len(parsed_profile_result["ncalls"])

        times = [
            parsed_profile_result["tottime"][i] for i in range(line_count)
            if "sqlite3" in parsed_profile_result["filename:lineno(function)"][i]
        ]

        total_time = sum([float(i) for i in times])
        total_times.append(total_time)

    print(total_times)


def memory_performance():
    result = stress_test_simple(400)
    print(result)


if __name__ == "__main__":
    memory_performance()
