"""
Deprecated Module for backwards compatibility
"""

import warnings

from qcodes.utils import QCoDeSDeprecationWarning

from ._AimTTi_PL_P import AimTTi, AimTTiChannel, NotKnownModel

warnings.warn(
    "The `qcodes._drivers.AimTTi.AimTTi_PL601P_channels` module is deprecated. "
    "Please import drivers from from `qcodes.instrument_drivers.AimTTi` instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
