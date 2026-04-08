"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""

import warnings

from qcodes.instrument_drivers.Keithley.Keithley_2450 import (
    Keithley2450,
    Keithley2450Buffer,
    Keithley2450Sense,
    Keithley2450Source,
    ParameterWithSetpointsCustomized,
    _SweepDict,
)
from qcodes.utils import QCoDeSDeprecationWarning

warnings.warn(
    "The `qcodes.instrument_drivers.tektronix.Keithley_2450` module is deprecated. "
    "Please import drivers from `qcodes.instrument_drivers.Keithley` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)

Buffer2450 = Keithley2450Buffer
Sense2450 = Keithley2450Sense
Source2450 = Keithley2450Source
