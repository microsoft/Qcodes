from qcodes.instrument_drivers.stanford_research._sg384 import SG384
from qcodes.utils.deprecate import _warn_deprecated_module

_warn_deprecated_module(__name__, "qcodes.instrument_drivers.stanford_research")

__all__ = ["SG384"]
