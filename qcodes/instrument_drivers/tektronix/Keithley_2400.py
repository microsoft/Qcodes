from qcodes.instrument.visa import VisaInstrument


class Keithley_2400(VisaInstrument):
    """
    TODO
    """
    def __init__(self, name, address, reset=False, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        self._modes = ['']

        # Add parameters to wrapper
        self.add_parameter('volt', get_cmd=':READ?',
                           get_parser=self.get_volt,
                           set_cmd='sour:volt:lev {:.5f};',
                           units='V')
        self.add_parameter('curr', get_cmd=':READ?',
                           get_parser=self.get_curr,
                           set_cmd='sour:curr:lev {:.5f};',
                           units='A')
        # One might want to initialize like this:
        # ':SOUR:VOLT:MODE FIX'
        # ':SENS:FUNC:ON "CURR"'

        self.connect_message()

    def get_curr(self, msg):
        return float(msg.split(',')[1])

    def get_volt(self, msg):
        return float(msg.split(',')[0])
