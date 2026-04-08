"""
Old alias for RigolDG4000 for backwards compatibility.
Will eventually be deprecated and removed.
"""

import warnings

from qcodes.utils import QCoDeSDeprecationWarning

from .Rigol_DG4000 import (
    RigolDG4000,
    clean_string,
    is_number,
    parse_multiple_outputs,
    parse_single_output,
    parse_string_output,
)

warnings.warn(
    "The `qcodes.instrument_drivers.rigol.DG4000` module is deprecated. "
    "Please import drivers from `qcodes.instrument_drivers.rigol` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)


class Rigol_DG4000(RigolDG4000):
    pass
