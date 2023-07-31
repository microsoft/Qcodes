import json
import logging
import math
import re
from typing import TYPE_CHECKING

import numpy as np

from qcodes.utils import NumpyJSONEncoder, deep_update, issue_deprecation_warning

try:
    from qcodes_loop.data.data_array import DataArray
    from qcodes_loop.data.format import Formatter
    from qcodes_loop.data.gnuplot_format import GNUPlotFormat
except ImportError as e:
    raise ImportError(
        "qcodes.data.gunplot_format is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e
issue_deprecation_warning(
    "qcodes.data.gunplot_format module", alternative="qcodes_loop.data.gunplot_format"
)
