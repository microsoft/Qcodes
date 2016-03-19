import time
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals
import numpy as np


class Weinschel_8320(VisaInstrument):
    '''
    QCodes driver for the stepped attenuator
    Weinschel is formerly known as Aeroflex/Weinschel
    '''

    def __init__(self, name, address):
        t0 = time.time()
        self.add_parameter('IDN',
                           get_cmd='*IDN?')
        self.add_parameter('attenuation', units='dB',
                           set_cmd='ATTN ALL {0:0=2d}',
                           get_cmd='ATTN? 1',
                           vals=vals.Enum(*np.arange(0, 60.1, 2).tolist()),
                           get_parser=float)

        super().__init__(name, address, terminator='\r')

        self.connect_message('IDN', t0)
