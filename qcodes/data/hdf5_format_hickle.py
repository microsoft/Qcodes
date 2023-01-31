import logging
from typing import TYPE_CHECKING

from qcodes.utils import deep_update, issue_deprecation_warning

try:
    from qcodes_loop.data.hdf5_format import HDF5Format
    from qcodes_loop.data.hdf5_format_hickle import HDF5FormatHickle
except ImportError as e:
    raise ImportError(
        "qcodes.data.hdf5_format_hickle is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e
issue_deprecation_warning(
    "qcodes.data.hdf5_format_hickle module",
    alternative="qcodes_loop.data.hdf5_format_hickle",
)
