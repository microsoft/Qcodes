"""
Live plotting in Jupyter notebooks
using the nbagg backend and matplotlib
"""
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import partial

import numpy as np
from numpy.ma import getmask, masked_invalid

import qcodes
from qcodes.utils import issue_deprecation_warning

try:
    from qcodes_loop.data.data_array import DataArray
    from qcodes_loop.plots.base import BasePlot
    from qcodes_loop.plots.qcmatplotlib import MatPlot
except ImportError as e:
    raise ImportError(
        "qcodes.plots.qcmatplotlib is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e
issue_deprecation_warning(
    "qcodes.plots.qcmatplotlib module", alternative="qcodes_loop.plots.qcmatplotlib"
)
