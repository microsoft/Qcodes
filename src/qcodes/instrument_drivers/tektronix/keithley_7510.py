"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""

from qcodes.instrument_drivers.Keithley.Keithley_7510 import (
    DataArray7510,
    GeneratedSetPoints,
    Keithley7510,
    Keithley7510Buffer,
    Keithley7510DigitizeSense,
    Keithley7510Sense,
)

Buffer7510 = Keithley7510Buffer
Sense7510 = Keithley7510Sense
DigitizeSense7510 = Keithley7510DigitizeSense
