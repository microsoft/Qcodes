# Instrument generating a timestamp
#

import time
from qcodes import Instrument


class TimeStampInstrument(Instrument):
    '''
    Instrument that generates a timestamp
    '''
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        # we need this to be a parameter (a function does not work with measure)
        self.add_parameter('timestamp', units='s', get_cmd=time.time, docstring='Timestamp based on number of seconds since epoch.')
        _ = self.timestamp.get()
        
    