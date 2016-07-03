from qcodes import VisaInstrument


class Keithley_2600(VisaInstrument):
    """
    channel: use channel 'a' or 'b'

    This is the qcodes driver for the Keithley_2600 Source-Meter series,
    tested with Keithley_2614B

    Status: beta-version.
        TODO:
        - Add all parameters that are in the manual
        - range and limit should be set according to mode
        - add ramping and such stuff

    """
    def __init__(self, name, address, channel, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)
        self._channel = channel

        self.add_parameter('volt', get_cmd='measure.v()',
                           get_parser=float, set_cmd='source.levelv={:.8f}',
                           label='Voltage',
                           units='V')
        self.add_parameter('curr', get_cmd='measure.i()',
                           get_parser=float, set_cmd='source.leveli={:.8f}',
                           label='Current',
                           units='A')
        self.add_parameter('mode',
                           get_cmd='source.func',
                           set_cmd='source.func={:d}',
                           val_mapping={'current': 0, 'voltage': 1})
        self.add_parameter('output',
                           get_cmd='source.output',
                           set_cmd='source.output={:d}',
                           val_mapping={'on':  1, 'off': 0})
        # Source range
        # needs get after set
        self.add_parameter('rangev',
                           get_cmd='source.rangev',
                           get_parser=float,
                           set_cmd='source.rangev={:.4f}',
                           units='V')
        # Measure range
        # needs get after set
        self.add_parameter('rangei',
                           get_cmd='source.rangei',
                           get_parser=float,
                           set_cmd='source.rangei={:.4f}',
                           units='A')
        # Compliance limit
        self.add_parameter('limitv',
                           get_cmd='source.limitv',
                           get_parser=float,
                           set_cmd='source.limitv={:.4f}',
                           units='V')
        # Compliance limit
        self.add_parameter('limiti',
                           get_cmd='source.limiti',
                           get_parser=float,
                           set_cmd='source.limiti={:.4f}',
                           units='A')

        self.connect_message()

    def get_idn(self):
        IDN = self.ask_raw('*IDN?')
        vendor, model, serial, firmware = map(str.strip, IDN.split(','))
        model = model[6:]

        IDN = {'vendor': vendor, 'model': model,
               'serial': serial, 'firmware': firmware}
        return IDN

    def reset(self):
        self.write('reset()')

    def ask(self, cmd):
        return super().ask('print(smu{:s}.{:s})'.format(self._channel, cmd))

    def write(self, cmd):
        super().write('smu{:s}.{:s}'.format(self._channel, cmd))
