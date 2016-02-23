import time
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals
import logging
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

    def run_tests(self):
        '''
        Imports the modules needed for running the test suite and runs
        some basic tests to verify that the instrument is working correctly
        '''
        import unittest
        from . import test_suite
        from importlib import reload
        reload(test_suite)
        test_suite.instr = self
        suite = unittest.TestLoader().loadTestsFromTestCase(
            test_suite.stepped_attenuator)
        unittest.TextTestRunner(verbosity=2).run(suite)

