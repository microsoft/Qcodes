from qcodes.instrument.visa import VisaInstrument
from qcodes.utils.validators import Numbers


class GS200(VisaInstrument):
    """
    This is the qcodes driver for the Yokogawa GS200 voltage and current source

    Args:
      name (str): What this instrument is called locally.
      address (str): The GPIB address of this instrument
      kwargs (dict): kwargs to be passed to VisaInstrument class

    TODO:(nataliejpg)
    - add current functionality (mode settings)
    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)

        self.add_parameter('voltage',
                           label='Voltage',
                           unit='V',
                           get_cmd=':SOURce:LEVel?',
                           set_cmd=':SOURce:LEVel:AUTO {:.4f}',
                           get_parser=float,
                           vals=Numbers(-10, 10))

        self.add_function('reset', call_cmd='*RST')

        self.initialise()
        self.connect_message()

    def initialise(self):
        self.write(':SYST:DISP ON')
        self.write(':SOUR:FUNC VOLT')
        self.write(':SOUR:PROT:CURR MIN')
        self.write(':OUTP:STAT ON')
