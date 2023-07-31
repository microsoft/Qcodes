"""DataSet class and factory functions."""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from copy import deepcopy
from traceback import format_exc
from typing import TYPE_CHECKING, Any, Callable, Dict

import numpy as np

from qcodes.utils import (
    DelegateAttributes,
    deep_update,
    full_class,
    issue_deprecation_warning,
)

try:
    from qcodes_loop.data.data_array import (
        DataArray,
        data_array_to_xarray_dictionary,
        xarray_data_array_dictionary_to_data_array,
    )
    from qcodes_loop.data.data_set import (
        DataSet,
        _PrettyPrintDict,
        dataset_to_xarray_dictionary,
        load_data,
        new_data,
        qcodes_dataset_to_xarray_dataset,
        xarray_dataset_to_qcodes_dataset,
        xarray_dictionary_to_dataset,
    )
    from qcodes_loop.data.gnuplot_format import GNUPlotFormat
    from qcodes_loop.data.io import DiskIO
    from qcodes_loop.data.location import FormatLocation
except ImportError as e:
    raise ImportError(
        "qcodes.data.data_set is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e

issue_deprecation_warning(
    "qcodes.data.data_set module", alternative="qcodes_loop.data.data_set"
)
