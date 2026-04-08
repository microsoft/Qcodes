"""
Old alias for RigolDG1062 for backwards compatibility.
Will eventually be deprecated and removed.
"""

import warnings

from qcodes.utils import QCoDeSDeprecationWarning

from .Rigol_DG1062 import RigolDG1062, RigolDG1062Burst, RigolDG1062Channel

warnings.warn(
    "The `qcodes.instrument_drivers.rigol.DG1062` module is deprecated. "
    "Please import drivers from `qcodes.instrument_drivers.rigol` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)


class DG1062(RigolDG1062):
    pass
