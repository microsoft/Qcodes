from typing import TYPE_CHECKING

from .private.DP8xx import RigolDP8xxBase

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs


class RigolDP832(RigolDP8xxBase):
    """
    This is the qcodes driver for the Rigol DP832(A) Power Supply
    """

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ):
        channel_ranges = [(30.0, 3.0), (30.0, 3.0), (5.0, 3.0)]

        ovp_ranges_std = [(0.01, 33.0), (0.01, 33.0), (0.01, 5.5)]
        ocp_ranges_std = [(0.001, 3.3), (0.001, 3.3), (0.001, 3.3)]

        ovp_ranges_precision = [(0.001, 33.0), (0.001, 33.0), (0.001, 5.5)]
        ocp_ranges_precision = [(0.001, 3.3), (0.001, 3.3), (0.001, 3.3)]

        ovp = (ovp_ranges_std, ovp_ranges_precision)
        ocp = (ocp_ranges_std, ocp_ranges_precision)

        super().__init__(name, address, channel_ranges, ovp, ocp, **kwargs)
