"""
Legacy module kept around for backwards compatibility reasons.
Will eventually be deprecated and removed

"""

import warnings

from qcodes.utils import QCoDeSDeprecationWarning

from .BaselSP983c import BaselSP983c

warnings.warn(
    "The `qcodes._drivers.basel.sp983c` module is deprecated. "
    "Please import drivers from from `qcodes.instrument_drivers.basel` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)


class SP983C(BaselSP983c):
    def get_idn(self) -> dict[str, str | None]:
        vendor = "Physics Basel"
        model = "SP 983(c)"
        serial = None
        firmware = None
        return {
            "vendor": vendor,
            "model": model,
            "serial": serial,
            "firmware": firmware,
        }
