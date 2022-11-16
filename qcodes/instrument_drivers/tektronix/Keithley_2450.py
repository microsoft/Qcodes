"""
Alias left for backwards compatibility.
Keithly drivers have moved to the Keithley module
"""
from qcodes.instrument_drivers.Keithley.Keithley_2450 import (
    Keithley2450,
    Keithley2450Buffer,
    Keithley2450Sense,
    Keithley2450Source,
    ParameterWithSetpointsCustomized,
    _SweepDict,
)

Buffer2450 = Keithley2450Buffer
Sense2450 = Keithley2450Sense
Source2450 = Keithley2450Source
