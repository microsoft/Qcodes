from .SD_common.SD_AWG import *
from .SD_common.SD_FPGA import *

class Keysight_M3201A_AWG(SD_AWG):
    """
    This is the qcodes driver for the Keysight M3201A AWG PXIe card.

    Args:
        name (str)      : name for this instrument, passed to the base instrument
        chassis (int)   : chassis number where the device is located
        slot (int)      : slot number where the device is plugged in
    """

    def __init__(self, name, chassis=1, slot=7, **kwargs):
        super().__init__(name, chassis, slot, channels=4, triggers=8, **kwargs)

class Keysight_M3201A_FPGA(SD_FPGA):
    """
    This is the qcodes driver for the Keysight M3201A AWG PXIe card's onboard
    FPGA.

    Args:
        name (str)      : name for this instrument, passed to the base instrument
        chassis (int)   : chassis number where the device is located
        slot (int)      : slot number where the device is plugged in
    """

    def __init__(self, name, chassis=1, slot=7, **kwargs):
        super().__init__(name, chassis, slot, **kwargs)
