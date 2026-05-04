"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""

from qcodes.instrument_drivers.Keithley.Keithley_6500 import (
    Keithley6500,
    Keithley6500CommandSetError,
    _parse_output_bool,
    _parse_output_string,
)

CommandSetError = Keithley6500CommandSetError

