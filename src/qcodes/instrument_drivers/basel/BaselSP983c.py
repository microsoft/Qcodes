from qcodes.instrument_drivers.basel._basel_sp983c import BaselSP983c
from qcodes.utils.deprecate import _warn_deprecated_module

_warn_deprecated_module(__name__, "qcodes.instrument_drivers.basel")

__all__ = ["BaselSP983c"]
