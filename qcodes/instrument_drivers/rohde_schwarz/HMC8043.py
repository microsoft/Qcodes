from .private.HMC804x import _RohdeSchwarzHMC804x
from qcodes.utils.deprecate import deprecate_moved_to_qcd


@deprecate_moved_to_qcd(alternative="qcodes_contrib_drivers.drivers.RohdeSchwarz.HMC8043.RohdeSchwarzHMC8043")
class RohdeSchwarzHMC8043(_RohdeSchwarzHMC804x):
    """
    This is the qcodes driver for the Rohde & Schwarz HMC8043 Power Supply
    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, num_channels=3, **kwargs)
