"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""

from typing_extensions import deprecated

from qcodes.instrument_drivers.Keithley.Keithley_2400 import Keithley2400
from qcodes.utils.deprecate import QCoDeSDeprecationWarning


@deprecated(
    "Keithley_2400 is deprecated. Please use qcodes.instrument_drivers.Keithley.Keithley2400 instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class Keithley_2400(Keithley2400):
    """
    Backwards compatibility alias for Keithley 2400 driver
    """

    pass
