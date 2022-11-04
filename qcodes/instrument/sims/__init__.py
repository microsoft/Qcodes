import sys
from importlib.abc import Traversable

if sys.version_info >= (3, 9):
    from importlib.resources import as_file, files
else:
    from importlib_resources import as_file, files


def get_sim_file_path(filename: str) -> Traversable:

    file = files("qcodes.instrument.sims") / filename
    return file
