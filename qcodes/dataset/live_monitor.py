"""
This module is intended to provide various methods to monitor in real time
data that is added to the data set
"""

from typing import Callable
import sys


def printer(names: list) -> Callable:
    """
    A simple function that will print inserted data to stdout in such a way
    that single line on the output prompt is refreshed at every call. This
    prevents the output prompt from being filled with clutter from previous
    iterations.

    Example:
    >>> import numpy as np
    >>> from qcodes.dataset.data_set import new_data_set, ParamSpec
    >>> specs=[ParamSpec("x", "numeric", unit='Hz'), ParamSpec("y", "numeric", unit='V')]
    >>> data_set = new_data_set("test", specs=specs)
    >>> sub_id = data_set.subscribe(printer(["x:.2", "y:.2e"]), state=[])
    >>> for x in np.linspace(100, 200, 150):
    >>>     y = np.random.randn()
    >>>     data_set.add_result({"x": x, "y": y})
    >>> # We shall see a single output line being refreshed

    Args
    ----
    names (list): List of strings

    Returns
    -------
    subscriber (Callable): A callable that can be used as a data set subscriber

    TODO
    ----
    Find out how we can make this multi line!
    """

    def split_format(txt, splitter):
        if splitter not in txt:
            return txt, "{}"

        splits = txt.split(splitter)
        if len(splits) > 2:
            raise ValueError(f"'{txt}' has invalid format")

        splits[1] = "{:" + splits[1] + "}"
        return splits

    fmt = [split_format(name, ":") for name in names]
    s_template = ", ".join(["{} = {}".format(*f) for f in fmt])

    def subscriber(results, length, state):
        s = s_template.format(*results[-1])
        sys.stdout.write("\r\x1b[K" + s.__str__())
        sys.stdout.flush()

    return subscriber
