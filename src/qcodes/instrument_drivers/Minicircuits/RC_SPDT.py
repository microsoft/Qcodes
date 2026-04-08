from typing_extensions import deprecated

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from ._minicircuits_rc_spdt import MiniCircuitsRCSPDT, MiniCircuitsRCSPDTChannel


@deprecated(
    "MC_channel is deprecated. Please use qcodes.instrument_drivers.Minicircuits.MiniCircuitsRCSPDTChannel instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class MC_channel(MiniCircuitsRCSPDTChannel):
    """
    Alias for backwards compatibility
    """

    pass


@deprecated(
    "RC_SPDT is deprecated. Please use qcodes.instrument_drivers.Minicircuits.MiniCircuitsRCSPDT instead.",
    category=QCoDeSDeprecationWarning,
    stacklevel=1,
)
class RC_SPDT(MiniCircuitsRCSPDT):
    """
    Alias for backwards compatibility
    """

    pass
