"""
Live plotting using pyqtgraph
"""
import logging
import warnings
from collections import deque, namedtuple
from typing import Deque, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pyqtgraph as pg
import pyqtgraph.multiprocess as pgmp
from pyqtgraph import QtGui
from pyqtgraph.graphicsItems.PlotItem.PlotItem import PlotItem
from pyqtgraph.multiprocess.remoteproxy import ClosedError, ObjectProxy

import qcodes
import qcodes.utils.qt_helpers
from qcodes.utils import issue_deprecation_warning

try:
    from qcodes_loop.plots.base import BasePlot
    from qcodes_loop.plots.colors import color_cycle, colorscales
    from qcodes_loop.plots.pyqtgraph import QtPlot, TransformState
except ImportError as e:
    raise ImportError(
        "qcodes.plots.pyqtgraph is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e
issue_deprecation_warning(
    "qcodes.plots.pyqtgraph", alternative="qcodes_loop.plots.pyqtgraph"
)
