"""
Old alias for RigolDS4000 for backwards compatibility.
Will eventually be deprecated and removed.
"""

from .Rigol_DS4000 import (
    RigolDS4000,
    RigolDS4000Channel,
    RigolDS4000TraceNotReady,
    ScopeArray,
)

TraceNotReady = RigolDS4000TraceNotReady


class DS4000(RigolDS4000):
    pass
