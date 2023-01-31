"""
This module has moved to the qcodes_loop package.
"""

from qcodes_loop.data.data_array import DataArray
from qcodes_loop.data.format import Formatter
from qcodes_loop.data.gnuplot_format import GNUPlotFormat
from qcodes_loop.data.hdf5_format import HDF5Format
from qcodes_loop.data.io import DiskIO
from qcodes_loop.data.location import FormatLocation

from qcodes.utils import issue_deprecation_warning

issue_deprecation_warning("qcodes.data", alternative="qcodes_loop.data")
