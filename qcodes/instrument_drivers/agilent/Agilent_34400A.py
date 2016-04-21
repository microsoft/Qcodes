# Agilent_34401A.py driver for Agilent 34401A DMM
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


from qcodes.utils.validators import Enum, Strings
from qcodes import VisaInstrument


class Agilent_34400A(VisaInstrument):
    '''
    This is the qcodes driver for the Agilent_34400A DMM Series,
    tested with Agilent_34401A

    Status: beta-version.
        TODO:
        - Add all parameters that are in the manual
        - Integration time does no value mapping
        - Need a clear_cmd thing in the parameter
        - Add labels

    '''
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        self.IDN = self.visa_handle.ask('*IDN?')

        vendor, model, serial, software = map(str.strip, self.IDN.split(','))
        self.model = model
        self.info = {'vendor': vendor, 'model': model,
                     'serial_number': serial, 'software_revision': software}

        # Async has to send 'INIT' and later ask for 'FETCH?'

        self.add_parameter('volt',
                           get_cmd='READ?',
                           label='Voltage',
                           get_parser=float,
                           units='V')
        self.add_parameter('volt_fetch',
                           get_cmd='FETCH?',
                           label='Voltage',
                           get_parser=float,
                           units='V')
        self.add_parameter('NPLC',
                           get_cmd='VOLT:NPLC?',
                           get_parser=float,
                           set_cmd='VOLT:NPLC {:f}',
                           vals=Enum(0.02, 0.2, 1, 10, 100))
        # For DC and resistance measurements, changing the number of digits
        # does more than just change the resolution of the multimeter. It also
        # changes the integration time!
        # Resolution Choices          Integration Time
        #   Fast 4 Digit                0.02 PLC
        #   * Slow 4 Digit              1 PLC
        #   Fast 5 Digit                0.2 PLC
        #   * Slow 5 Digit (default)    10 PLC
        #   * Fast 6 Digit              10 PLC
        #   Slow 6 Digit                100 PLC
        self.add_parameter('resolution',
                           get_cmd='VOLT:DC:RES?',
                           get_parser=float,
                           set_cmd='VOLT:DC:RES {:.7f}',
                           vals=Enum(3e-07, 1e-06, 3e-06, 1e-05, 1e-04),
                           units='V')
        # Integration Time    Resolutionc
        self.add_parameter('integration_time',
                           get_cmd='VOLT:DC:RES?',
                           get_parser=float,
                           set_cmd='VOLT:DC:RES {:f}',
                           # vals=Enum(0.02,0.2,1,10,100),
                           units='NPLC',
                           val_mapping={0.02: 0.0001,
                                        0.2:  0.00001,
                                        1:    0.000003,
                                        10:   0.000001,
                                        100:  0.0000003})
        self.add_parameter('terminals',
                           get_cmd='ROUT:TERM?')
        self.add_parameter('range_auto',
                           get_cmd='VOLT:RANG:AUTO?',
                           get_parser=self._onoff_parser,
                           set_cmd='VOLT:RANG:AUTO {:d}',
                           val_mapping={'ON': 1,
                                        'OFF': 0,
                                        1: 1,
                                        0: 0})
        self.add_parameter('range',
                           get_cmd='SENS:VOLT:DC:RANG?',
                           get_parser=float,
                           set_cmd='SENS:VOLT:DC:RANG {:f}',
                           vals=Enum(0.1, 1.0, 10.0, 100.0, 1000.0))

        if self.model in ['34401A']:
            self.add_parameter('display_text',
                               get_cmd='DISP:TEXT?',
                               set_cmd='DISP:TEXT "{}"',
                               vals=Strings())
        elif self.model in ['34410A', '34411A']:
            self.add_parameter('display_text',
                               get_cmd='DISP:WIND1:TEXT?',
                               set_cmd='DISP:WIND1:TEXT "{}"',
                               vals=Strings())
            self.add_parameter('display_text_2',
                               get_cmd='DISP:WIND2:TEXT?',
                               set_cmd='DISP:WIND2:TEXT "{}"',
                               vals=Strings())

    def init_measurement(self):
        self.write('INIT')

    def display_clear(self):
        if self.model in ['34401A']:
            self.write('DISP:WIND:TEXT:CLE')
            self.write('DISP:WIND:STAT 1')
        elif self.model in ['34410A', '34411A']:
            self.write('DISP:WIND1:TEXT:CLE')
            self.write('DISP:WIND1:STAT 1')
            self.write('DISP:WIND2:TEXT:CLE')
            self.write('DISP:WIND2:STAT 1')

    def reset(self):
        self.write('*RST')

    def _onoff_parser(self, msg):
        if msg == '0':
            return 'OFF'
        elif msg == '1':
            return 'ON'
        return None
