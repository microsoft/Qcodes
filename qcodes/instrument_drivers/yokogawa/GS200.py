from qcodes.instrument.visa import VisaInstrument
from qcodes.utils.validators import Numbers


class GS200(VisaInstrument):

    def __init__(self, name, address, reset=False, **kwargs):
        super().__init__(name, address, **kwargs)

        # Add parameters to wrapper
        self.add_parameter('voltage',
						   label='Voltage',
                           units='V',
                           get_cmd=':SOURce:LEVel?',
                           set_cmd=':SOURce:LEVel:AUTO {:.4f}',
                           get_parser=float,
                           vals=Numbers(-10, 10))
                           
        self.initialise()
        self.connect_message()
        
    def initialise(self):
        self.write('*RST')
        self.write(':SYST:DISP ON')
        self.write(':SOUR:FUNC VOLT')
        self.write(':SOUR:PROT:CURR MIN')
        self.write(':OUTP:STAT ON')
        self.voltage.set(0)
