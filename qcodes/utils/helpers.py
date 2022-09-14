"""
Module left for backwards compatibility.
Please do not import from this in any new code
"""
import logging
from contextlib import contextmanager

# for backwards compatibility since this module used
# to contain logic that would abstract between yaml
# libraries.
from ruamel.yaml import YAML

from qcodes.loops import tprint, wait_secs
from qcodes.parameters.named_repr import named_repr
from qcodes.parameters.permissive_range import permissive_range
from qcodes.parameters.sequence_helpers import is_sequence, is_sequence_of
from qcodes.parameters.sweep_values import make_sweep
from qcodes.parameters.val_mapping import create_on_off_val_mapping
from qcodes.tests.common import compare_dictionaries

from .abstractmethod import qcodes_abstractmethod as abstractmethod
from .attribute_helpers import (
    DelegateAttributes,
    attribute_set_to,
    checked_getattr,
    strip_attrs,
)
from .deep_update_utils import deep_update
from .full_class import full_class
from .function_helpers import is_function
from .json_utils import NumpyJSONEncoder
from .partial_utils import partial_with_docstring
from .path_helpers import QCODES_USER_PATH_ENV, get_qcodes_path, get_qcodes_user_path
from .qt_helpers import foreground_qt_window
from .spyder_utils import add_to_spyder_UMR_excludelist


# on longer in used but left for backwards compatibility until
# module is removed.
def warn_units(class_name: str, instance: object) -> None:
    logging.warning('`units` is deprecated for the `' + class_name +
                    '` class, use `unit` instead. ' + repr(instance))


# TODO these functions need a place
import builtins
import sys
import time
from pprint import pprint

import numpy as np

from qcodes.configuration.config import DotDict


def using_ipython() -> bool:
    """Check if code is run from IPython (including jupyter notebook/lab)"""
    return hasattr(builtins, "__IPYTHON__")


def directly_executed_from_cell(level: int = 1) -> bool:
    """Test if this function is called directly from an IPython cell
    The IPython prompt is also valid.

    Args:
         level: Difference in frames from IPython cell/prompt to check.
            Since the check is executed from this function, the default level is 1.

    Returns:
        True if directly run from IPython cell/prompt, False otherwise

    Examples:
        These examples should be run in a notebook cell.

        >>> directly_executed_from_cell()
        ... True

        >>> def wrap_function(**kwargs):
        >>>     return directly_executed_from_cell(**kwargs)
        >>> wrap_function()
        ... False
        >>> wrap_function(level=2)
        ... True

    """
    if level < 1:
        raise SyntaxError("Level must be 1 or higher")

    frame = sys._getframe(level)
    return "_" in frame.f_locals


def get_last_input_cells(cells=3):
    """
    Get last input cell. Note that get_last_input_cell.globals must be set to
    the ipython globals
    Returns:
        last cell input if successful, else None
    """
    global In
    if "In" in globals() or hasattr(builtins, "In"):
        return In[-cells:]
    else:
        logging.warning("No input cells found")


def get_exponent(val):
    prefactors = [
        (9, "G"),
        (6, "M"),
        (3, "k"),
        (0, ""),
        (-3, "m"),
        (-6, "u"),
        (-9, "n"),
    ]
    for exponent, prefactor in prefactors:
        if val >= np.power(10.0, exponent):
            return exponent, prefactor
    else:
        return prefactors[-1]


class PerformanceTimer:
    max_records = 100

    def __init__(self):
        self.timings = DotDict()

    def __getitem__(self, key):
        val = self.timings.__getitem__(key)
        return self._timing_to_str(val)

    def __repr__(self):
        return pprint.pformat(self._timings_to_str(self.timings), indent=2)

    def clear(self):
        self.timings.clear()

    def _timing_to_str(self, val):
        mean_val = np.mean(val)
        exponent, prefactor = get_exponent(mean_val)
        factor = np.power(10.0, exponent)

        return f"{mean_val / factor:.3g}+-{np.abs(np.std(val))/factor:.3g} {prefactor}s"

    def _timings_to_str(self, d: dict):

        timings_str = DotDict()
        for key, val in d.items():
            if isinstance(val, dict):
                timings_str[key] = self._timings_to_str(val)
            else:
                timings_str[key] = self._timing_to_str(val)

        return timings_str

    @contextmanager
    def record(self, key, val=None):
        if isinstance(key, str):
            timing_list = self.timings.setdefault(key, [])
        elif isinstance(key, (list)):
            *parent_keys, subkey = key
            d = self.timings.create_dicts(*parent_keys)
            timing_list = d.setdefault(subkey, [])
        else:
            raise ValueError("Key must be str or list/tuple")

        if val is not None:
            timing_list.append(val)
        else:
            t0 = time.perf_counter()
            yield
            t1 = time.perf_counter()
            timing_list.append(t1 - t0)

        # Optionally remove oldest elements
        for _ in range(len(timing_list) - self.max_records):
            timing_list.pop(0)
