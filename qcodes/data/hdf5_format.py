import json
import logging
import os
from typing import TYPE_CHECKING

import numpy as np

import qcodes as qc
from qcodes.utils import NumpyJSONEncoder, deep_update, issue_deprecation_warning

try:
    from qcodes_loop.data.data_array import DataArray
    from qcodes_loop.data.format import Formatter
    from qcodes_loop.data.hdf5_format import (
        HDF5Format,
        HDF5FormatMetadata,
        _encode_to_utf8,
        str_to_bool,
    )
except ImportError as e:
    raise ImportError(
        "qcodes.data.hdf5_format is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e
issue_deprecation_warning(
    "qcodes.data.hdf5_format module", alternative="qcodes_loop.data.hdf5_format"
)
