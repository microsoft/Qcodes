from typing import TYPE_CHECKING

from .private.DP8xx import RigolDP8xxBase

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs


class RigolDP821(RigolDP8xxBase):
    """
    This is the qcodes driver for the Rigol DP821(A) Power Supply
    """

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ):
        channel_ranges = [
            (60.0, 1.0),
            (8.0, 10.0),
        ]

        ovp_ranges_std = [(0.01, 66.0), (0.01, 8.8)]
        ocp_ranges_std = [(0.01, 1.1), (0.01, 11)]

        ovp_ranges_precision = [(0.001, 66.0), (0.001, 8.8)]
        ocp_ranges_precision = [(0.001, 1.1), (0.001, 11)]

        ovp = (ovp_ranges_std, ovp_ranges_precision)
        ocp = (ocp_ranges_std, ocp_ranges_precision)

        super().__init__(name, address, channel_ranges, ovp, ocp, **kwargs)
