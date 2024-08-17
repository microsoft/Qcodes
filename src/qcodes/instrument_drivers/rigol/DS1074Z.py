"""
Old alias for RigolDS1074Z for backwards compatibility.
Will eventually be deprecated and removed.
"""

from .Rigol_DS1074Z import RigolDS1074Z, RigolDS1074ZChannel


class DS1074Z(RigolDS1074Z):
    pass
