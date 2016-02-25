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
        super().__init__(name, address)
        self.visa_handle.read_termination = '\r'
        self.add_parameter('IDN',
                           get_cmd='*IDN?')
        self.add_parameter('attenuation', units='dB',
                           set_cmd='ATTN ALL {0:0=2d}',
                           get_cmd='ATTN? 1',
                           vals=vals.Enum(*np.arange(0, 60.1, 2).tolist()),
                           get_parser=float)
        t1 = time.time()
        print('Connected to: ',
              self.get('IDN').replace(',', ', ').replace('\n', ' '),
              'in %.2fs' % (t1-t0))
