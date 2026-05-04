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

__all__ = [
    "FrequencyAxis",
    "KeysightN9030B",
    "KeysightN9030BPhaseNoiseMode",
    "KeysightN9030BSpectrumAnalyzerMode",
    "PhaseNoiseMode",
    "SpectrumAnalyzerMode",
    "Trace",
]
