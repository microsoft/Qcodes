# Instrument generating a timestamp
#

import time
from qcodes import Instrument


class TimeStampInstrument(Instrument):
    '''
    Instrument that generates a timestamp
    '''
    def __init__(self, name):
        super().__init__(name)

        # we need this to be a parameter (a function does not work with measure)
        self.add_parameter('timestamp', units='s', get_cmd=time.time)
        #self.add_function('timestamp', units='s', call_cmd=time.time, return_parser=float)

        _ = self.timestamp.get()
        
    