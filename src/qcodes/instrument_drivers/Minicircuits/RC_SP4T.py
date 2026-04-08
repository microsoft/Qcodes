from typing_extensions import deprecated

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from ._minicircuits_rc_sp4t import MiniCircuitsRCSP4T, MiniCircuitsRCSP4TChannel


@deprecated(
    "MC_channel is deprecated. Please use qcodes.instrument_drivers.Minicircuits.MiniCircuitsRCSP4TChannel instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class MC_channel(MiniCircuitsRCSP4TChannel):
    """
    Alias for backwards compatibility
    """

    pass


@deprecated(
    "RC_SP4T is deprecated. Please use qcodes.instrument_drivers.Minicircuits.MiniCircuitsRCSP4T instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class RC_SP4T(MiniCircuitsRCSP4T):
    """
    Alias for backwards compatibility
    """

    pass
