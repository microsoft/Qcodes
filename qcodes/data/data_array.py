import collections.abc
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

import logging

from qcodes.utils import DelegateAttributes, full_class, issue_deprecation_warning

_LOG = logging.getLogger(__name__)

try:
    from qcodes_loop.data.data_array import (
        DataArray,
        data_array_to_xarray_dictionary,
        xarray_data_array_dictionary_to_data_array,
    )
except ImportError as e:
    raise ImportError(
        "qcodes.data.data_array is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e

issue_deprecation_warning(
    "qcodes.data.data_array module", alternative="qcodes_loop.data.data_array"
)
