"""
Old alias for RigolDS4000 for backwards compatibility.
Will eventually be deprecated and removed.
"""

import warnings

from qcodes.utils import QCoDeSDeprecationWarning

from .Rigol_DS4000 import (
    RigolDS4000,
    RigolDS4000Channel,
    RigolDS4000TraceNotReady,
    ScopeArray,
)

warnings.warn(
    "The `qcodes.instrument_drivers.rigol.DS4000` module is deprecated. "
    "Please import drivers from `qcodes.instrument_drivers.rigol` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)

TraceNotReady = RigolDS4000TraceNotReady


class DS4000(RigolDS4000):
    pass
