from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

if sys.version_info >= (3, 9):
    from importlib.abc import Traversable
    from importlib.resources import as_file, files
else:
    from importlib_resources import as_file, files
    from importlib_resources.abc import Traversable


def get_sim_file_path(filename: str) -> Traversable:

    file = files("qcodes.instrument.sims") / filename
    return file


@contextmanager
def sims_visalib(filename: str) -> Iterator[str]:
    traversable_file = get_sim_file_path(filename=filename)
    with as_file(traversable_file) as file:
        yield f"{str(file)}@sim"
