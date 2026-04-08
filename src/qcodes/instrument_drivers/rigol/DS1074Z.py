"""
Old alias for RigolDS1074Z for backwards compatibility.
Will eventually be deprecated and removed.
"""

import warnings

from qcodes.utils import QCoDeSDeprecationWarning

from .Rigol_DS1074Z import RigolDS1074Z, RigolDS1074ZChannel

warnings.warn(
    "The `qcodes.instrument_drivers.rigol.DS1074Z` module is deprecated. "
    "Please import drivers from `qcodes.instrument_drivers.rigol` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)


class DS1074Z(RigolDS1074Z):
    pass
