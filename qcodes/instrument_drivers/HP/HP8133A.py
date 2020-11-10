from typing import Any

from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers, Ints, Enum


class HP8133A(VisaInstrument):
    """
    This is the code for Hewlett Packard 8133A Pulse Generator
    """

    def __init__(
            self,
            name: str,
            address: str,
            reset: bool = False,
            **kwargs: Any):
        super().__init__(name, address,  terminator='\n', **kwargs)

        self.add_parameter(name='frequency',
                           label='Frequency',
                           unit='Hz',
                           get_cmd='FREQ?',
                           set_cmd='FREQ {}',
                           get_parser=float,
                           vals=Numbers(min_value=31.3e6,
                                        max_value=3.5e9))
        self.add_parameter(name='period',
                           label='Period',
                           unit='s',
                           get_cmd='PER?',
                           set_cmd='PER {}',
                           get_parser=float,
                           vals=Numbers(min_value=286e-12,
                                        max_value=31.949e-9))
        self.add_parameter(name='phase',
                           label='Phase',
                           unit='deg',
                           get_cmd='PHAS?',
                           set_cmd='PHAS {}',
                           get_parser=float,
                           vals=Numbers(min_value=-3.6e3,
                                        max_value=3.6e3))
        self.add_parameter(name='duty_cycle',
                           label='Duty cycle',
                           unit='%',
                           get_cmd='DCYC?',
                           set_cmd='DCYC {}',
                           get_parser=float,
                           vals=Numbers(min_value=0,
                                        max_value=100))
        self.add_parameter(name='delay',
                           label='Delay',
                           unit='s',
                           get_cmd='DEL?',
                           set_cmd='DEL {}',
                           get_parser=float,
                           vals=Numbers(min_value=-5e-9,
                                        max_value=5e-9))
        self.add_parameter(name='width',
                           label='Width',
                           unit='s',
                           get_cmd='WIDT?',
                           set_cmd='WIDT {}',
                           get_parser=float,
                           vals=Numbers(min_value=1e-12,
                                        max_value=10.5e-9))
        self.add_parameter(name='amplitude',
                           label='Amplitude',
                           unit='V',
                           get_cmd='VOLT?',
                           set_cmd='VOLT {}',
                           get_parser=float,
                           vals=Numbers(min_value=0.1,
                                        max_value=3.3))
        self.add_parameter(name='amplitude_offset',
                           label='Offset',
                           unit='V',
                           get_cmd='VOLT:OFFS?',
                           set_cmd='VOLT:OFFS {}',
                           get_parser=float,
                           vals=Numbers(min_value=-2.95,
                                        max_value=3.95))
        self.add_parameter(name='output',
                           label='Output',
                           get_cmd='OUTP?',
                           set_cmd='OUTP {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1})

        # resets amplitude and offset each time user connects
        self.amplitude(0.1)
        self.amplitude_offset(0)

        self.add_function('reset', call_cmd='*RST')
        self.connect_message()
