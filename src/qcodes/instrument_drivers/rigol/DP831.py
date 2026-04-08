"""
Old alias for RigolDP831 for backwards compatibility.
Will eventually be deprecated and removed.
"""

import warnings

from qcodes.utils import QCoDeSDeprecationWarning

from .Rigol_DP831 import RigolDP831

warnings.warn(
    "The `qcodes.instrument_drivers.rigol.DP831` module is deprecated. "
    "Please import drivers from `qcodes.instrument_drivers.rigol` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
