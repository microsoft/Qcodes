from functools import partial
import logging
from typing import Union

from qcodes import VisaInstrument, validators as vals
from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument.base import Instrument
from qcodes.instrument_drivers.Keysight.private.error_handling import \
    KeysightErrorQueueMixin

log = logging.getLogger(__name__)


# This is to be the grand unified driver superclass for
# The Keysight/Agilent/HP Waveform generators of series
# 33200, 33500, and 33600


class OutputChannel(InstrumentChannel):
    """
    Class to hold the output channel of a waveform generator
    """
    def __init__(self, parent: Instrument, name: str, channum: int) -> None:
        """
        Args:
            parent: The instrument to which the channel is
                attached.
            name: The name of the channel
            channum: The number of the channel in question (1-2)
        """
        super().__init__(parent, name)

        def val_parser(parser: type, inputstring: str) -> Union[float,int]:
            """
            Parses return values from instrument. Meant to be used when a query
            can return a meaningful finite number or a numeric representation
            of infinity

            Args:
                parser: Either int or float, what to return in finite
                    cases
                inputstring: The raw return value
            """

            inputstring = inputstring.strip()

            if float(inputstring) == 9.9e37:
                output = float('inf')
            else:
                output = float(inputstring)
                if parser == int:
                    output = parser(output)

            return output

        self.model = self._parent.model

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

        max_freq = self._parent._max_freqs[self.model]
        self.add_parameter('frequency',
                           label='Channel {} frequency'.format(channum),
                           set_cmd='SOURce{}:FREQuency {{}}'.format(channum),
                           get_cmd='SOURce{}:FREQuency?'.format(channum),
                           get_parser=float,
                           unit='Hz',
                           # TODO: max. freq. actually really tricky
                           vals=vals.Numbers(1e-6, max_freq)
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

        self.add_parameter('pulse_width',
                           label="Channel {} pulse width".format(channum),
                           set_cmd='SOURce{}:FUNCtion:PULSE:WIDTh {{}}'.format(channum),
                           get_cmd='SOURce{}:FUNCtion:PULSE:WIDTh?'.format(channum),
                           get_parser=float,
                           unit='S')

        # TRIGGER MENU
        self.add_parameter('trigger_source',
                           label='Channel {} trigger source'.format(channum),
                           set_cmd='TRIGger{}:SOURce {{}}'.format(channum),
                           get_cmd='TRIGger{}:SOURce?'.format(channum),
                           vals=vals.Enum('IMM', 'EXT', 'TIM', 'BUS'),
                           get_parser=str.rstrip,
                           )

        self.add_parameter('trigger_slope',
                           label='Channel {} trigger slope'.format(channum),
                           set_cmd='TRIGger{}:SLOPe {{}}'.format(channum),
                           get_cmd='TRIGger{}:SLOPe?'.format(channum),
                           vals=vals.Enum('POS', 'NEG'),
                           get_parser=str.rstrip
                           )

        # Older models do not have all the fancy trigger options
        if self._parent.model[2] in ['5', '6']:
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
                               unit='s')

            self.add_parameter('trigger_timer',
                               label='Channel {} trigger timer'.format(channum),
                               set_cmd='TRIGger{}:TIMer {{}}'.format(channum),
                               get_cmd='TRIGger{}:TIMer?'.format(channum),
                               vals=vals.Numbers(1e-6, 8000),
                               get_parser=float)

        # TODO: trigger level doesn't work remotely. Why?

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
                           get_cmd='SOURce{}:BURSt:INTernal:PERiod?'.format(channum),
                           unit='s',
                           vals=vals.Numbers(1e-6, 8e3),
                           get_parser=float,
                           docstring=('The burst period is the time '
                                      'between the starts of consecutive '
                                      'bursts when trigger is immediate.')
                           )


class SyncChannel(InstrumentChannel):
    """
    Class to hold the sync output. Has very few parameters for
    single channel instruments
    """

    def __init__(self, parent, name):

        super().__init__(parent, name)

        self.add_parameter('output',
                           label='Sync output state',
                           set_cmd='OUTPut:SYNC {}',
                           get_cmd='OUTPut:SYNC?',
                           val_mapping={'ON': 1, 'OFF': 0},
                           vals=vals.Enum('ON', 'OFF')
                           )

        if parent.num_channels == 2:

            self.add_parameter('source',
                               label='Source of sync function',
                               set_cmd='OUTPut:SYNC:SOURce {}',
                               get_cmd='OUTPut:SYNC:SOURce?',
                               val_mapping={1: 'CH1', 2: 'CH2'},
                               vals=vals.Enum(1, 2))


class WaveformGenerator_33XXX(KeysightErrorQueueMixin, VisaInstrument):
    """
    QCoDeS driver for the Keysight/Agilent 33XXX series of
    waveform generators
    """

    def __init__(self, name, address, silent=False, **kwargs):
        """
        Args:
            name (string): The name of the instrument used internally
                by QCoDeS. Must be unique.
            address (string): The VISA resource name.
            silent (Optional[bool]): If True, no connect message is printed.
        """

        super().__init__(name, address, terminator='\n', **kwargs)
        self.model = self.IDN()['model']

        #######################################################################
        # Here go all model specific traits

        # TODO: Fill out this dict with all models
        no_of_channels = {'33210A': 1,
                          '33250A': 1,
                          '33511B': 1,
                          '33512B': 2,
                          '33522B': 2,
                          '33622A': 2
                          }

        self._max_freqs = {'33210A': 10e6,
                           '33511B': 20e6,
                           '33512B': 20e6,
                           '33250A': 80e6,
                           '33522B': 30e6,
                           '33622A': 120e6}

        self.num_channels = no_of_channels[self.model]

        for i in range(1, self.num_channels+1):
            channel = OutputChannel(self, 'ch{}'.format(i), i)
            self.add_submodule('ch{}'.format(i), channel)

        sync = SyncChannel(self, 'sync')
        self.add_submodule('sync', sync)

        self.add_function('force_trigger', call_cmd='*TRG')

        self.add_function('sync_channel_phases', call_cmd='PHAS:SYNC')

        if not silent:
            self.connect_message()
