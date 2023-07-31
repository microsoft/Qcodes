"""
This module has moved to the qcodes_loop package.
"""
from qcodes.utils import issue_deprecation_warning

try:
    from qcodes_loop.data.data_array import DataArray
    from qcodes_loop.data.format import Formatter
    from qcodes_loop.data.gnuplot_format import GNUPlotFormat
    from qcodes_loop.data.hdf5_format import HDF5Format
    from qcodes_loop.data.io import DiskIO
    from qcodes_loop.data.location import FormatLocation
except ImportError as e:
    raise ImportError(
        "qcodes.data is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e

issue_deprecation_warning("qcodes.data module", alternative="qcodes_loop.data")
