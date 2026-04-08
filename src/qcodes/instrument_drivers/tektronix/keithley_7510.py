"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""

import warnings

from qcodes.instrument_drivers.Keithley.Keithley_7510 import (
    DataArray7510,
    GeneratedSetPoints,
    Keithley7510,
    Keithley7510Buffer,
    Keithley7510DigitizeSense,
    Keithley7510Sense,
)
from qcodes.utils import QCoDeSDeprecationWarning

warnings.warn(
    "The `qcodes.instrument_drivers.tektronix.keithley_7510` module is deprecated. "
    "Please import drivers from `qcodes.instrument_drivers.Keithley` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)

Buffer7510 = Keithley7510Buffer
Sense7510 = Keithley7510Sense
DigitizeSense7510 = Keithley7510DigitizeSense
