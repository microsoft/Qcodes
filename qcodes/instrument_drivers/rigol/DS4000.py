"""
Old alias for RigolDS4000 for backwards compatibility.
Will eventually be deprecated and removed.
"""
from .RigolDS4000 import (
    RigolDS4000,
    RigolDS4000Channel,
    RigolDS4000TraceNotReady,
    ScopeArray,
)


class DS4000(RigolDS4000):
    pass
