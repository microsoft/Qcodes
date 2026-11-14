from qcodes.instrument_drivers.basel._basel_sp983 import BaselSP983
from qcodes.utils.deprecate import _warn_deprecated_module

_warn_deprecated_module(__name__, "qcodes.instrument_drivers.basel")

__all__ = ["BaselSP983"]
