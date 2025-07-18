"""
Module left for backwards compatibility. Will be deprecated and removed eventually.
"""

import warnings

from qcodes.utils import QCoDeSDeprecationWarning

from .HP_8133A import HP8133A

warnings.warn(
    "The `qcodes._drivers.HP.HP8133A` module is deprecated. "
    "Please import drivers from from `qcodes.instrument_drivers.HP` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
