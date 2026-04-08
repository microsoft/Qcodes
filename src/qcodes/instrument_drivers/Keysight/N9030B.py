from typing_extensions import deprecated

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from .Keysight_N9030B import (
    FrequencyAxis,
    KeysightN9030B,
    KeysightN9030BPhaseNoiseMode,
    KeysightN9030BSpectrumAnalyzerMode,
    Trace,
)

SpectrumAnalyzerMode = KeysightN9030BSpectrumAnalyzerMode
"""Alias for backwards compatibility"""

PhaseNoiseMode = KeysightN9030BPhaseNoiseMode
"""Alias for backwards compatibility"""


@deprecated(
    "N9030B is deprecated. Please use qcodes.instrument_drivers.Keysight.KeysightN9030B instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class N9030B(KeysightN9030B):
    """
    Alias for backwards compatibility
    """
