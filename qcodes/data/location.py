"""Standard location_provider class(es) for creating DataSet locations."""
import re
import string
from datetime import datetime
from typing import cast

import qcodes
from qcodes.utils import issue_deprecation_warning

try:
    from qcodes_loop.data.location import FormatLocation, SafeFormatter
except ImportError as e:
    raise ImportError(
        "qcodes.data.location is deprecated and has moved to "
        "the package `qcodes_loop`. Please install qcodes_loop directly or "
        "with `pip install qcodes[loop]"
    ) from e
issue_deprecation_warning(
    "qcodes.data.location module", alternative="qcodes_loop.data.location"
)
