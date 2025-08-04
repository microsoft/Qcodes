"""
Module left for backwards compatibility. Please do not depend on this in any
new code.
"""

import warnings

from qcodes.dataset.snapshot_utils import diff_param_values_by_id
from qcodes.metadatable import Metadatable
from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from .snapshot_helpers import (
    ParameterDiff,
    ParameterKey,
    Snapshot,
    diff_param_values,
    extract_param_values,
)

warnings.warn(
    "The `qcodes.utils.metadata` module is deprecated. "
    "Please consult the api documentation at https://microsoft.github.io/Qcodes/api/index.html for alternatives.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
