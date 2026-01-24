"""
Module left for backwards compatibility. Will be deprecated and removed eventually.
"""

import warnings

from qcodes.utils import QCoDeSDeprecationWarning

from .HP_8753D import HP8753D

warnings.warn(
    "The `qcodes._drivers.HP.HP8753D` module is deprecated. "
    "Please import drivers from from `qcodes.instrument_drivers.HP` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
