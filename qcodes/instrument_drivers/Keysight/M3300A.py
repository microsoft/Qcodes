###################################################################################
#                                                                                 #
#                               Driver file for M3300A                            #
#                                                                                 #
###################################################################################
#                                                                                 #
# Written by: Mark Johnson                                                        #
# Also see: http://www.keysight.com/en/pd-2747490-pn-M3300A                       #
#                                                                                 #
###################################################################################

try:
    from .SD_common.SD_AWG import SD_AWG
    from .SD_common.SD_DIG import SD_DIG
except ImportError:
    raise ImportError('To use the M3300A driver, install the keysight module')

class M3300A_AWG(SD_AWG):
    """ Driver for the AWG of the Keysight M3300A card.

    Args:
        name (str)    : name for this instrument, passed to the base instrument
        chassis (int) : chassis number where the device is located
        slot (int)    : slot number where the device is plugged in
    Example:
        AWG = AWG('M3300A')
    """
    def __init__(self, name, chassis=1, slot=8, **kwargs):
        super().__init__(name, chassis=1, slot=8, channels=4, triggers=8, **kwargs)

class M3300A_DIG(SD_DIG):
    """ Driver for the digitizer of the keysight M3300A card.

    Args:
        name (str)    : name for this instrument, passed to the base instrument
        chassis (int) : chassis number where the device is located
        slot (int)    : slot number where the device is plugged in

    Example:
        DIG  = DIG('M3300A')
    """
    def __init__(self, name, chassis=1, slot=8, **kwargs):
        super().__init__(name, chassis, slot, channels=8, triggers=8, **kwargs)
