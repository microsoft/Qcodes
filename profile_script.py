import cProfile

from qcodes import ParamSpec, new_data_set, new_experiment
import numpy as np
import re

new_experiment("profile", "profile")


def capture(f):
    """
    Decorator to capture standard output
    """
    def captured(*args, **kwargs):
        import sys
        from io import StringIO

        backup = sys.stdout

        try:
            sys.stdout = StringIO()     # capture output
            f(*args, **kwargs)
            out = sys.stdout.getvalue() # release output
        finally:
            sys.stdout.close()   # close the stream
            sys.stdout = backup  # restore original stdout

        return out  # captured output wrapped in a string

    return captured


def parse_profile_result(profile_result):

    lines = profile_result.split("\n")
    header = None
    parse_result = {}

    for line in lines:

        if "ncalls" in line:
            header = ["ncalls",  "tottime",  "percall_1", "cumtime",  "percall_2", "filename:lineno(function)"]
            parse_result = {h: [] for h in header}
            continue

        if header is None or line == "":
            continue

        line_data = re.split("(?<![a-z'])\s+", line.strip())

        for key, data in zip(header, line_data):
            parse_result[key].append(data)

    return parse_result


def stress_test_simple(sz):

    data_set = new_data_set("stress_test_simple")
    t1 = ParamSpec('t', 'real', label='time', unit='s')
    x = ParamSpec('x', 'real', label='voltage', unit='v', depends_on=[t1])

    data_set.add_parameter(t1)
    data_set.add_parameter(x)

    t_values = np.linspace(-1, 1, sz)

    results = [{"t": t, "x": 2 * t**2 + 1} for t in t_values]
    data_set.add_results(results)


@capture
def run_profiler(sz):
    cProfile.run("stress_test_simple({sz})".format(sz=sz))


def main():

    szes = [10, 50, 100, 150, 200, 250, 300, 350, 400]
    total_times = []

    for sz in szes:
        profile_result = run_profiler(sz)
        parsed_profile_result = parse_profile_result(profile_result)

        line_count = len(parsed_profile_result["ncalls"])
        times = [parsed_profile_result["tottime"][i] for i in range(line_count) if "sqlite3" in parsed_profile_result["filename:lineno(function)"][i]]
        total_time = sum([float(i) for i in times])
        print("total time = ", total_time)
        total_times.append(total_time)

    print(total_times)

if __name__ == "__main__":
    main()
