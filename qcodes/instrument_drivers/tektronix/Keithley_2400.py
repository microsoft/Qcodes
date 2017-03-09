from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers


class Keithley_2400(VisaInstrument):
    """
    Driver for the Keithley 2400 multimeter.

    Args:
      name
      address
      reset (bool): optional
    """

    def __init__(self, name, address, reset=False, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        self.add_parameter(name='voltage',
                           set_cmd='SOURCE:VOLTAGE:LEVEL {:.4f}',
                           get_cmd='SOURCE:VOLTAGE:LEVEL?',
                           get_parser=float,
                           vals=Numbers(-10, 10)
                           )

        if reset:
            self.initialise()
        self.connect_message()

    def initialise(self):
        self.write('SOURCE:FUNCTION:MODE VOLTAGE')
        self.write('INIT')
