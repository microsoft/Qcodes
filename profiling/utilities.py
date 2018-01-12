import re


def capture_stdout(f):
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