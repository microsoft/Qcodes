from qcodes import VisaInstrument

from qcodes.instrument_drivers.Keysight.keysightb1500.message_builder import \
    MessageBuilder


class KeysightB1500(VisaInstrument):
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\r\n', **kwargs)

        self.mb = MessageBuilder()

        # TODO do a UNT query to determine Modules. generate a module object for
        # each found module.

    def reset(self):
        """Performs an instrument reset.

        Does not reset error queue!
        """
        self.write('*RST')

    def get_status(self):
        return self.ask('*STB?')

