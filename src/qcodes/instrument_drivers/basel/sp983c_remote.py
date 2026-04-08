"""
Legacy module kept around for backwards compatibility reasons.
Will eventually be deprecated and removed

"""

import warnings

from qcodes.utils import QCoDeSDeprecationWarning

from .BaselSP983a import BaselSP983a as SP983A

warnings.warn(
    "The `qcodes._drivers.basel.sp983c_remote` module is deprecated. "
    "Please import drivers from from `qcodes.instrument_drivers.basel` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
