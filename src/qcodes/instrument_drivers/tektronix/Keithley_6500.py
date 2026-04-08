"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""

from typing_extensions import deprecated

from qcodes.instrument_drivers.Keithley.Keithley_6500 import (
    Keithley6500,
    Keithley6500CommandSetError,
    _parse_output_bool,
    _parse_output_string,
)
from qcodes.utils.deprecate import QCoDeSDeprecationWarning

CommandSetError = Keithley6500CommandSetError


@deprecated(
    "Keithley_6500 is deprecated. Please use qcodes.instrument_drivers.Keithley.Keithley6500 instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class Keithley_6500(Keithley6500):
    """
    Alias left for backwards compatibility. Will eventually be deprecated and removed.
    """

    pass
