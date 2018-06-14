# from qcodes import validators as validator
# from functools import partial
#
# from .SD_common.SD_Module import *
from .SD_common.SD_AWG import SD_AWG


class Keysight_M3201A(SD_AWG):
    """
    This is the qcodes driver for the Keysight M3201A AWG PXIe card

    Args:
        name (str): name for this instrument, passed to the base instrument
        chassis (int): chassis number where the device is located
        slot (int): slot number where the device is plugged in
    """

    def __init__(self, name, chassis=1, slot=7, **kwargs):
        super().__init__(name, chassis, slot, channels=4, triggers=8, **kwargs)
