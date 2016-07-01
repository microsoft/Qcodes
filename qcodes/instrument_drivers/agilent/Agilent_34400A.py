from qcodes.utils.validators import Enum, Strings
from qcodes import VisaInstrument


class Agilent_34400A(VisaInstrument):
    """
    This is the qcodes driver for the Agilent_34400A DMM Series,
    tested with Agilent_34401A

    Status: beta-version.
        TODO:
        - Add all parameters that are in the manual
        - Integration time does no value mapping
        - Need a clear_cmd thing in the parameter
        - Add labels

    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        idn = self.IDN.get()
        self.model = idn['model']
        # Async has to send 'INIT' and later ask for 'FETCH?'

        self.add_parameter('volt',
                           get_cmd='READ?',
                           label='Voltage',
                           get_parser=float,
                           units='V')
        self.add_parameter('fetch',
                           get_cmd='FETCH?',
                           label='Voltage',
                           get_parser=float,
                           units='V',
                           snapshot_get=False,
                           docstring=('Reads the data you asked for, i.e.'
                                      'after an `init_measurement()` you can '
                                      'read the data with fetch.\n'
                                      'Do not call this when you didn\'t ask '
                                      'for data in the first place!'))
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
                           # TODO: does this work for get?
                           val_mapping={0.02: 0.0001,
                                        0.2:  0.00001,
                                        1:    0.000003,
                                        10:   0.000001,
                                        100:  0.0000003})
        self.add_parameter('terminals',
                           get_cmd='ROUT:TERM?')
        self.add_parameter('range_auto',
                           get_cmd='VOLT:RANG:AUTO?',
                           set_cmd='VOLT:RANG:AUTO {:d}',
                           val_mapping={'on': 1,
                                        'off': 0})
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

        self.connect_message()

    def clear_errors(self):
        while True:
            err = self.ask('SYST:ERR?')
            if 'No error' in err:
                return
            print(err)

    def init_measurement(self):
        self.write('INIT')

    def display_clear(self):
        if self.model in ['34401A']:
            lines = ['WIND']
        elif self.model in ['34410A', '34411A']:
            lines = ['WIND1', 'WIND2']
        else:
            raise ValueError('unrecognized model: ' + str(self.model))

        for line in lines:
            self.write('DISP:' + line + ':TEXT:CLE')
            self.write('DISP:' + line + ':STAT 1')

    def reset(self):
        self.write('*RST')
