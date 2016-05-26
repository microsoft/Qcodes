# Keithley_2600.py driver for Keithley 2600 Source-Meter series
#
# The MIT License (MIT)
# Copyright (c) 2016 Merlin von Soosten <merlin.von.soosten@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in theSoftware without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from qcodes import VisaInstrument


class Keithley_2600(VisaInstrument):
    '''
    channel: use channel 'a' or 'b'

    This is the qcodes driver for the Keithley_2600 Source-Meter series,
    tested with Keithley_2614B

    Status: beta-version.
        TODO:
        - Add all parameters that are in the manual
        - range and limit should be set according to mode
        - add ramping and such stuff

    '''
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
                           get_parser=self._mode_parser,
                           set_cmd='source.func={:d}',
                           val_mapping={'current': 0, 'curr': 0, 'AMPS': 0,
                                        'voltage': 1, 'volt': 1, 'VOLT': 1})
        self.add_parameter('output',
                           get_cmd='source.output',
                           get_parser=self._output_parser,
                           set_cmd='source.output={:d}',
                           val_mapping={'on':  1, 'ON':  1,
                                        'off': 0, 'OFF': 0})
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

    def get_idn(self):
        IDN = self.ask_direct('*IDN?')
        vendor, model, serial, firmware = map(str.strip, IDN.split(','))
        model = model[6:]

        IDN = {'vendor': vendor, 'model': model,
               'serial': serial, 'firmware': firmware}
        return IDN

    def _mode_parser(self, msg):
        if msg[0] == '0':
            return 'current'
        elif msg[0] == '1':
            return 'voltage'
        return None

    def _output_parser(self, msg):
        if msg[0] == '0':
            return 'OFF'
        elif msg[0] == '1':
            return 'ON'
        return None

    def reset(self):
        self.write('reset()')

    def ask_direct(self, cmd):
        return self.visa_handle.ask(cmd)

    def ask(self, cmd):
        return self.visa_handle.ask('print(smu{:s}.{:s})'.format(self._channel, cmd))

    def write(self, cmd):
        super().write('smu{:s}.{:s}'.format(self._channel, cmd))
