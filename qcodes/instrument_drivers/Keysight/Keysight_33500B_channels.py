from qcodes import VisaInstrument, validators as vals
from functools import partial
from qcodes.instrument.channel import InstrumentChannel, ChannelList
import logging

log = logging.getLogger(__name__)


class KeysightChannel(InstrumentChannel):
    """


    """
    def __init__(self, parent, name, channum):
        """
        Args:
            parent (Instrument): The instrument to which the channel is
                attached.
            name (str): The name of the channel
            channum (int): The number of the channel in question (1-2)
        """
        super().__init__(parent, name)

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

        self.add_parameter('function_type',
                           label='Channel {} function type'.format(channum),
                           set_cmd='SOURce{}:FUNCtion {{}}'.format(channum),
                           get_cmd='SOURce{}:FUNCtion?'.format(channum),
                           get_parser=str.rstrip,
                           vals=vals.Enum('SIN', 'SQU', 'TRI', 'RAMP',
                                          'PULS', 'PRBS', 'NOIS', 'ARB',
                                          'DC')
                           )

        self.add_parameter('frequency_mode',
                           label='Channel {} frequency mode'.format(channum),
                           set_cmd='SOURce{}:FREQuency:MODE {{}}'.format(channum),
                           get_cmd='SOURce{}:FREQuency:MODE?'.format(channum),
                           get_parser=str.rstrip,
                           vals=vals.Enum('CW', 'LIST', 'SWEEP', 'FIXED')
                           )

        self.add_parameter('frequency',
                           label='Channel {} frequency'.format(channum),
                           set_cmd='SOURce{}:FREQuency {{}}'.format(channum),
                           get_cmd='SOURce{}:FREQuency?'.format(channum),
                           get_parser=float,
                           unit='Hz',
                           # TODO: max. freq. actually really tricky
                           vals=vals.Numbers(1e-6, 30e6)
                           )

        self.add_parameter('phase',
                           label='Channel {} phase'.format(channum),
                           set_cmd='SOURce{}:PHASe {{}}'.format(channum),
                           get_cmd='SOURce{}:PHASe?'.format(channum),
                           get_parser=float,
                           unit='deg',
                           vals=vals.Numbers(0, 360)
                           )
        self.add_parameter('amplitude_unit',
                           label='Channel {} amplitude unit'.format(channum),
                           set_cmd='SOURce{}:VOLTage:UNIT {{}}'.format(channum),
                           get_cmd='SOURce{}:VOLTage:UNIT?'.format(channum),
                           vals=vals.Enum('VPP', 'VRMS', 'DBM'),
                           get_parser=str.rstrip
                           )

        self.add_parameter('amplitude',
                           label='Channel {} amplitude'.format(channum),
                           set_cmd='SOURce{}:VOLTage {{}}'.format(channum),
                           get_cmd='SOURce{}:VOLTage?'.format(channum),
                           unit='',  # see amplitude_unit
                           get_parser=float)

        self.add_parameter('offset',
                           label='Channel {} voltage offset'.format(channum),
                           set_cmd='SOURce{}:VOLTage:OFFSet {{}}'.format(channum),
                           get_cmd='SOURce{}:VOLTage:OFFSet?'.format(channum),
                           unit='V',
                           get_parser=float
                           )
        self.add_parameter('output',
                           label='Channel {} output state'.format(channum),
                           set_cmd='OUTPut{} {{}}'.format(channum),
                           get_cmd='OUTPut{}?'.format(channum),
                           val_mapping={'ON': 1, 'OFF': 0}
                           )

        self.add_parameter('ramp_symmetry',
                           label='Channel {} ramp symmetry'.format(channum),
                           set_cmd='SOURce{}:FUNCtion:RAMP:SYMMetry {{}}'.format(channum),
                           get_cmd='SOURce{}:FUNCtion:RAMP:SYMMetry?'.format(channum),
                           get_parser=float,
                           unit='%',
                           vals=vals.Numbers(0, 100)
                           )

        # TRIGGER MENU
        self.add_parameter('trigger_source',
                           label='Channel {} trigger source'.format(channum),
                           set_cmd='TRIGger{}:SOURce {{}}'.format(channum),
                           get_cmd='TRIGger{}:SOURce?'.format(channum),
                           vals=vals.Enum('IMM', 'EXT', 'TIM', 'BUS'),
                           get_parser=str.rstrip,
                           )

        self.add_parameter('trigger_count',
                           label='Channel {} trigger count'.format(channum),
                           set_cmd='TRIGger{}:COUNt {{}}'.format(channum),
                           get_cmd='TRIGger{}:COUNt?'.format(channum),
                           vals=vals.Ints(1, 1000000),
                           get_parser=partial(val_parser, int)
                           )

        self.add_parameter('trigger_delay',
                           label='Channel {} trigger delay'.format(channum),
                           set_cmd='TRIGger{}:DELay {{}}'.format(channum),
                           get_cmd='TRIGger{}:DELay?'.format(channum),
                           vals=vals.Numbers(0, 1000),
                           get_parser=float,
                           unit='s'
                           )

        # TODO: trigger level doesn't work remotely. Why?

        self.add_parameter('trigger_slope',
                           label='Channel {} trigger slope'.format(channum),
                           set_cmd='TRIGger{}:SLOPe {{}}'.format(channum),
                           get_cmd='TRIGger{}:SLOPe?'.format(channum),
                           vals=vals.Enum('POS', 'NEG'),
                           get_parser=str.rstrip
                           )

        self.add_parameter('trigger_timer',
                           label='Channel {} trigger timer'.format(channum),
                           set_cmd='TRIGger{}:TIMer {{}}'.format(channum),
                           get_cmd='TRIGger{}:TIMer?'.format(channum),
                           vals=vals.Numbers(1e-6, 8000),
                           get_parser=float
                           )

        # output menu
        self.add_parameter('output_polarity',
                           label='Channel {} output polarity'.format(channum),
                           set_cmd='OUTPut{}:POLarity {{}}'.format(channum),
                           get_cmd='OUTPut{}:POLarity?'.format(channum),
                           get_parser=str.rstrip,
                           vals=vals.Enum('NORM', 'INV')
                           )
        # BURST MENU
        self.add_parameter('burst_state',
                           label='Channel {} burst state'.format(channum),
                           set_cmd='SOURce{}:BURSt:STATe {{}}'.format(channum),
                           get_cmd='SOURce{}:BURSt:STATe?'.format(channum),
                           val_mapping={'ON': 1, 'OFF': 0},
                           vals=vals.Enum('ON', 'OFF')
                           )

        self.add_parameter('burst_mode',
                           label='Channel {} burst mode'.format(channum),
                           set_cmd='SOURce{}:BURSt:MODE {{}}'.format(channum),
                           get_cmd='SOURce{}:BURSt:MODE?'.format(channum),
                           get_parser=str.rstrip,
                           val_mapping={'N Cycle': 'TRIG', 'Gated': 'GAT'},
                           vals=vals.Enum('N Cycle', 'Gated')
                           )

        self.add_parameter('burst_ncycles',
                           label='Channel {} burst no. of cycles'.format(channum),
                           set_cmd='SOURce{}:BURSt:NCYCles {{}}'.format(channum),
                           get_cmd='SOURce{}:BURSt:NCYCLes?'.format(channum),
                           get_parser=partial(val_parser, int),
                           vals=vals.MultiType(vals.Ints(1),
                                               vals.Enum('MIN', 'MAX',
                                                         'INF'))
                           )

        self.add_parameter('burst_phase',
                           label='Channel {} burst start phase'.format(channum),
                           set_cmd='SOURce{}:BURSt:PHASe {{}}'.format(channum),
                           get_cmd='SOURce{}:BURSt:PHASe?'.format(channum),
                           vals=vals.Numbers(-360, 360),
                           unit='degrees',
                           get_parser=float
                           )

        self.add_parameter('burst_polarity',
                           label='Channel {} burst gated polarity'.format(channum),
                           set_cmd='SOURce{}:BURSt:GATE:POLarity {{}}'.format(channum),
                           get_cmd='SOURce{}:BURSt:GATE:POLarity?'.format(channum),
                           vals=vals.Enum('NORM', 'INV')
                           )

        self.add_parameter('burst_int_period',
                           label=('Channel {}'.format(channum) +
                                  ' burst internal period'),
                           set_cmd='SOURce{}:BURSt:INTernal:PERiod {{}}'.format(channum),
                           get_cmd='SOURce{}:BURSt:INTernal:PERiod'.format(channum),
                           unit='s',
                           vals=vals.Numbers(1e-6, 8e3),
                           get_parser=float,
                           docstring=('The burst period is the time '
                                      'between the starts of consecutive '
                                      'bursts when trigger is immediate.')
                           )


class Keysight_33500B_Channels(VisaInstrument):
    """
    QCoDeS driver for the Keysight 33500B Waveform Generator using QCoDeS
    channels
    """

    def __init__(self, name, address, silent=False, **kwargs):
        """
        Args:
            name (string): The name of the instrument used internally
                by QCoDeS. Must be unique.
            address (string): The VISA resource name.
            silent (Optional[bool]): If True, no connect message is printed.
        """

        super().__init__(name, address, **kwargs)

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

        channels = ChannelList(self, "Channels", KeysightChannel,
                               snapshotable=False)
        for i in range(1, 3):
            channel = KeysightChannel(self, 'ch{}'.format(i), i)
            channels.append(channel)
        channels.lock()
        self.add_submodule('channels', channels)

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

        self.add_function('sync_channel_phases', call_cmd='PHAS:SYNC')

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
