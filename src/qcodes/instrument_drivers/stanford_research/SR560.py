from qcodes.instrument_drivers.stanford_research._sr560 import (
    SR560,
    SR560VoltageParameter,
)
from qcodes.utils.deprecate import _warn_deprecated_module

_warn_deprecated_module(__name__, "qcodes.instrument_drivers.stanford_research")

VoltageParameter = SR560VoltageParameter

__all__ = ["SR560", "VoltageParameter"]
