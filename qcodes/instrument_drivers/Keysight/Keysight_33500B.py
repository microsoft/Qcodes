import warnings

from qcodes import VisaInstrument, validators as vals
from functools import partial
import logging

log = logging.getLogger(__name__)


class Keysight_33500B(VisaInstrument):
    """
    QCoDeS driver for the Keysight 33500B Waveform Generator
    """

    def __init__(self, name, address, silent=False, **kwargs):
        """
        Args:
            name (string): The name of the instrument used internally
                by QCoDeS. Must be unique.
            address (string): The VISA resource name.
            silent (Optional[bool]): If True, no connect message is printed.
        """

        warnings.warn("This driver is old and will be removed "
                      "from QCoDeS soon. Please use the "
                      "WaveformGenerator_33XXX from the file "
                      "instrument_drivers/Keysight/KeysightAgilent_33XXX"
                      " instead.", UserWarning)

        super().__init__(name, address, **kwargs)

        # convenient little helper functions
        def setcmd(channel, setting):
            return 'SOURce{}:'.format(channel) + setting + ' {}'

        def getcmd(channel, setting):
            return 'SOURce{}:'.format(channel) + setting + '?'

        def errorparser(rawmssg):
            """
            Parses the error message.

            Args:
                rawmssg (str): The raw return value of 'SYSTem:ERRor?'

            Returns:
                tuple (int, str): The error code and the error message.
            """
            code = int(rawmssg.split(',')[0])
            mssg = rawmssg.split(',')[1].strip().replace('"', '')

            return code, mssg

        def val_parser(parser, inputstring):
            """
            Parses return values from instrument. Meant to be used when a query
            can return a meaningful finite number or a numeric representation
            of infinity

            Args:
                inputstring (str): The raw return value
                parser (type): Either int or float, what to return in finite
                    cases
            """

            inputstring = inputstring.strip()

            if float(inputstring) == 9.9e37:
                output = float('inf')
            else:
                output = float(inputstring)
                if parser == int:
                    output = parser(output)

            return output

        for chan in [1, 2]:

            self.add_parameter('ch{}_function_type'.format(chan),
                               label='Channel {} function type'.format(chan),
                               set_cmd=setcmd(chan, 'FUNCtion'),
                               get_cmd=getcmd(chan, 'FUNCtion'),
                               vals=vals.Enum('SIN', 'SQU', 'TRI', 'RAMP',
                                              'PULS', 'PRBS', 'NOIS', 'ARB',
                                              'DC')
                               )

            self.add_parameter('ch{}_frequency_mode'.format(chan),
                               label='Channel {} frequency mode'.format(chan),
                               set_cmd=setcmd(chan, 'FREQuency:MODE'),
                               get_cmd=getcmd(chan, 'FREQuency:MODE'),
                               vals=vals.Enum('CW', 'LIST', 'SWEEP', 'FIXED')
                               )

            self.add_parameter('ch{}_frequency'.format(chan),
                               label='Channel {} frequency'.format(chan),
                               set_cmd=setcmd(chan, 'FREQuency'),
                               get_cmd=getcmd(chan, 'FREQUency'),
                               get_parser=float,
                               unit='Hz',
                               # TODO: max. freq. actually really tricky
                               vals=vals.Numbers(1e-6, 30e6)
                               )

            self.add_parameter('ch{}_phase'.format(chan),
                               label='Channel {} phase'.format(chan),
                               set_cmd=setcmd(chan, 'PHASe'),
                               get_cmd=getcmd(chan, 'PHASe'),
                               get_parser=float,
                               unit='deg',
                               vals=vals.Numbers(0, 360)
                               )
            self.add_parameter('ch{}_amplitude_unit'.format(chan),
                               label='Channel {} amplitude unit'.format(chan),
                               set_cmd=setcmd(chan, 'VOLTage:UNIT'),
                               get_cmd=getcmd(chan, 'VOLTage:UNIT'),
                               vals=vals.Enum('VPP', 'VRMS', 'DBM')
                               )

            self.add_parameter('ch{}_amplitude'.format(chan),
                               label='Channel {} amplitude'.format(chan),
                               set_cmd=setcmd(chan, 'VOLTage'),
                               get_cmd=getcmd(chan, 'VOLTage'),
                               unit='',  # see amplitude_unit
                               get_parser=float)

            self.add_parameter('ch{}_offset'.format(chan),
                               label='Channel {} voltage offset'.format(chan),
                               set_cmd=setcmd(chan, 'VOLTage:OFFSet'),
                               get_cmd=getcmd(chan, 'VOLTage:OFFSet'),
                               unit='V',
                               get_parser=float
                               )
            self.add_parameter('ch{}_output'.format(chan),
                               label='Channel {} output state'.format(chan),
                               set_cmd='OUTPut{}'.format(chan) + ' {}',
                               get_cmd='OUTPut{}?'.format(chan),
                               val_mapping={'ON': 1, 'OFF': 0}
                               )

            self.add_parameter('ch{}_output_polarity'.format(chan),
                               label='Channel {} output polarity'.format(chan),
                               set_cmd='OUTPut{}:POL'.format(chan) + ' {}',
                               get_cmd='OUTPut{}:POL?'.format(chan),
                               get_parser=str.strip,
                               vals=vals.Enum('NORM', 'INV')
                               )

            self.add_parameter('ch{}_ramp_symmetry'.format(chan),
                               label='Channel {} ramp symmetry'.format(chan),
                               set_cmd=setcmd(chan, 'FUNCtion:RAMP:SYMMetry'),
                               get_cmd=getcmd(chan, 'FUNCtion:RAMP:SYMMetry'),
                               get_parser=float,
                               vals=vals.Numbers(0, 100)
                               )

            # TRIGGER MENU
            self.add_parameter('ch{}_trigger_source'.format(chan),
                               label='Channel {} trigger source'.format(chan),
                               set_cmd='TRIGger{}:SOURce {}'.format(chan, '{}'),
                               get_cmd='TRIGger{}:SOURce?'.format(chan),
                               vals=vals.Enum('IMM', 'EXT', 'TIM', 'BUS')
                               )

            self.add_parameter('ch{}_trigger_count'.format(chan),
                               label='Channel {} trigger count'.format(chan),
                               set_cmd='TRIGger{}:COUNt {}'.format(chan, '{}'),
                               get_cmd='TRIGger{}:COUNt?'.format(chan),
                               vals=vals.Ints(1, 1000000),
                               get_parser=partial(val_parser, int)
                               )

            self.add_parameter('ch{}_trigger_delay'.format(chan),
                               label='Channel {} trigger delay'.format(chan),
                               set_cmd='TRIGger{}:DELay {}'.format(chan, '{}'),
                               get_cmd='TRIGger{}:DELay?'.format(chan),
                               vals=vals.Numbers(0, 1000),
                               get_parser=float,
                               unit='s'
                               )

            # TODO: trigger level doesn't work remotely. Why?

            self.add_parameter('ch{}_trigger_slope'.format(chan),
                               label='Channel {} trigger slope'.format(chan),
                               set_cmd='TRIGger{}:SLOPe {}'.format(chan, '{}'),
                               get_cmd='TRIGger{}:SLOPe?'.format(chan),
                               vals=vals.Enum('POS', 'NEG')
                               )

            self.add_parameter('ch{}_trigger_timer'.format(chan),
                               label='Channel {} trigger timer'.format(chan),
                               set_cmd='TRIGger{}:TIMer {}'.format(chan, '{}'),
                               get_cmd='TRIGger{}:TIMer?'.format(chan),
                               vals=vals.Numbers(1e-6, 8000),
                               get_parser=float
                               )

            # BURST MENU
            self.add_parameter('ch{}_burst_state'.format(chan),
                               label='Channel {} burst state'.format(chan),
                               set_cmd=setcmd(chan, 'BURSt:STATe'),
                               get_cmd=getcmd(chan, 'BURSt:STATe'),
                               val_mapping={'ON': 1, 'OFF': 0},
                               vals=vals.Enum('ON', 'OFF')
                               )

            self.add_parameter('ch{}_burst_mode'.format(chan),
                               label='Channel {} burst mode'.format(chan),
                               set_cmd=setcmd(chan, 'BURSt:MODE'),
                               get_cmd=getcmd(chan, 'BURSt:MODE'),
                               val_mapping={'N Cycle': 'TRIG', 'Gated': 'GAT'},
                               vals=vals.Enum('N Cycle', 'Gated')
                               )

            self.add_parameter('ch{}_burst_ncycles'.format(chan),
                               label='Channel {} burst no. of cycles'.format(chan),
                               set_cmd=setcmd(chan, 'BURSt:NCYCles'),
                               get_cmd=getcmd(chan, 'BURSt:NCYCLes'),
                               get_parser=partial(val_parser, int),
                               vals=vals.MultiType(vals.Ints(1),
                                                   vals.Enum('MIN', 'MAX',
                                                             'INF'))
                               )

            self.add_parameter('ch{}_burst_phase'.format(chan),
                               label='Channel {} burst start phase'.format(chan),
                               set_cmd=setcmd(chan, 'BURSt:PHASe'),
                               get_cmd=getcmd(chan, 'BURSt:PHASe'),
                               vals=vals.Numbers(-360, 360),
                               unit='degrees',
                               get_parser=float
                               )

            self.add_parameter('ch{}_burst_polarity'.format(chan),
                               label='Channel {} burst gated polarity'.format(chan),
                               set_cmd=setcmd(chan, 'BURSt:GATE:POLarity'),
                               get_cmd=getcmd(chan, 'BURSt:GATE:POLarity'),
                               vals=vals.Enum('NORM', 'INV')
                               )

            self.add_parameter('ch{}_burst_int_period'.format(chan),
                               label=('Channel {}'.format(chan) +
                                      ' burst internal period'),
                               set_cmd=setcmd(chan, 'BURSt:INTernal:PERiod'),
                               get_cmd=getcmd(chan, 'BURSt:INTernal:PERiod'),
                               unit='s',
                               vals=vals.Numbers(1e-6, 8e3),
                               get_parser=float,
                               docstring=('The burst period is the time '
                                          'between the starts of consecutive '
                                          'bursts when trigger is immediate.')
                               )

        self.add_parameter('sync_source',
                           label='Source of sync function',
                           set_cmd='OUTPut:SYNC:SOURce {}',
                           get_cmd='OUTPut:SYNC:SOURce?',
                           val_mapping={1: 'CH1', 2: 'CH2'},
                           vals=vals.Enum(1, 2)
                           )

        self.add_parameter('sync_output',
                           label='Sync output state',
                           set_cmd='OUTPut:SYNC {}',
                           get_cmd='OUTPut:SYNC?',
                           val_mapping={'ON': 1, 'OFF': 0},
                           vals=vals.Enum('ON', 'OFF')
                           )


        self.add_parameter('error',
                           label='Error message',
                           get_cmd='SYSTem:ERRor?',
                           get_parser=errorparser
                           )

        self.add_function('force_trigger', call_cmd='*TRG')

        if not silent:
            self.connect_message()

    def flush_error_queue(self, verbose=True):
        """
        Clear the instrument error queue.

        Args:
            verbose (Optional[bool]): If true, the error messages are printed.
                Default: True.
        """

        log.debug('Flushing error queue...')

        err_code, err_message = self.error()
        log.debug('    {}, {}'.format(err_code, err_message))
        if verbose:
            print(err_code, err_message)

        while err_code != 0:
            err_code, err_message = self.error()
            log.debug('    {}, {}'.format(err_code, err_message))
            if verbose:
                print(err_code, err_message)

        log.debug('...flushing complete')
