from qcodes.instrument_drivers.stanford_research._sr860 import SR860
from qcodes.utils.deprecate import _warn_deprecated_module

_warn_deprecated_module(__name__, "qcodes.instrument_drivers.stanford_research")

__all__ = ["SR860"]
