"""
Old alias for RigolDG1062 for backwards compatibility.
Will eventually be deprecated and removed.
"""

from .Rigol_DG1062 import RigolDG1062, RigolDG1062Burst, RigolDG1062Channel


class DG1062(RigolDG1062):
    pass
