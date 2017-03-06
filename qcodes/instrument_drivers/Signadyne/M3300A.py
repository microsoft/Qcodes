###################################################################################
#                                                                                 #
#                               Driver file for M3300A                            #
#                                                                                 #
###################################################################################
#                                                                                 #
# CQC2T                                                                           #
#                                                                                 #
# Written by: Mark Johnson                                                        #
# Also see: https://www.signadyne.com/en/products/hardware/generators-digitizers/ #
#                                                                                 #
###################################################################################

import numpy as np
import ctypes as ct
from functools import partial
from qcodes.utils.validators import Enum, Numbers, Anything
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
try:
    import signadyne_common.SD_AWG as SD_AWG
    import signadyne_common.SD_DIG as SD_DIG
except ImportError:
    raise ImportError('To use the M3300A driver, install the Signadyne module')

class M3300A(SD_DIG, SD_AWG):
    def __init__(self, name, cardid='', **kwargs):
        """ Driver for the Signadyne M3300A card.

        Example:

            Example usage for acquisition with channel 2 using an external trigger
            that triggers multiple times with trigger mode HIGH::

                m3300A = M3300A(name='M3300A')

        Todo:
          A lot.

        """
        super(SD_DIG, self).__init__(8)
        super(SD_AWG, self).__init__(8)

