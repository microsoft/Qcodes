"""
Old alias for RigolDG4000 for backwards compatibility.
Will eventually be deprecated and removed.
"""

from .Rigol_DG4000 import (
    RigolDG4000,
    clean_string,
    is_number,
    parse_multiple_outputs,
    parse_single_output,
    parse_string_output,
)


class Rigol_DG4000(RigolDG4000):
    pass
