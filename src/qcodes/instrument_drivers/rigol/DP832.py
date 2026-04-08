"""
Old alias for RigolDP832 for backwards compatibility.
Will eventually be deprecated and removed.
"""

import warnings

from qcodes.utils import QCoDeSDeprecationWarning

from .Rigol_DP832 import RigolDP832

warnings.warn(
    "The `qcodes.instrument_drivers.rigol.DP832` module is deprecated. "
    "Please import drivers from `qcodes.instrument_drivers.rigol` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
