"""
Old alias for RigolDP821 for backwards compatibility.
Will eventually be deprecated and removed.
"""

import warnings

from qcodes.utils import QCoDeSDeprecationWarning

from .Rigol_DP821 import RigolDP821

warnings.warn(
    "The `qcodes.instrument_drivers.rigol.DP821` module is deprecated. "
    "Please import drivers from `qcodes.instrument_drivers.rigol` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
