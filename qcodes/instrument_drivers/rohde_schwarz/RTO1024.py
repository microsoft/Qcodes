# All manual references are to R&S RTO Digital Oscilloscope User Manual
# for firmware 3.65, 2017

import numpy as np
from time import sleep
import warnings
from distutils.version import LooseVersion

from qcodes import Instrument
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ArrayParameter


class ScopeTrace(ArrayParameter):

    def __init__(self, name: str, instrument: VisaInstrument,
                 channel: int) -> None:
        super().__init__(name=name,
                         shape=(1,),
                         label='Voltage',  # TODO: Is this sometimes dbm?
                         unit='V',
                         setpoint_names=('Time',),
                         setpoint_labels=('Time',),
                         setpoint_units=('s',),
                         docstring='Holds scope trace')
        self.channel = channel
        self._instrument = instrument

    def prepare_trace(self) -> None:
        """
        Prepare the scope for returning data, calculate the setpoints
        """
        pass


class ScopeChannel(InstrumentChannel):
    """
    Class to hold an input channel of the scope.

    Exposes: state, coupling, ground, scale, range, position, offset,
    invert, bandwidth, impedance, overload
    """

    def __init__(self, parent: Instrument, name: str, channum: int) -> None:
        """
        Args:
            parent: The instrument to which the channel is attached
            name: The name of the channel
            channum: The number of the channel in question. Must match the
                actual number as used by the instrument (1..4)
        """

        if channum not in [1, 2, 3, 4]:
            raise ValueError('Invalid channel number! Must be 1, 2, 3, or 4.')

        self.channum = channum

        super().__init__(parent, name)

        self.add_parameter('state',
                           label='Channel {} state'.format(channum),
                           get_cmd='CHANnel{}:STATe?'.format(channum),
                           set_cmd='CHANnel{}:STATE {{}}'.format(channum),
                           vals=vals.Enum('ON', 'OFF'),
                           docstring='Switches the channel on or off')

        self.add_parameter('coupling',
                           label='Channel {} coupling'.format(channum),
                           get_cmd='CHANnel{}:COUPling?'.format(channum),
                           set_cmd='CHANnel{}:COUPling {{}}'.format(channum),
                           vals=vals.Enum('DC', 'DCLimit', 'AC'),
                           docstring=('Selects the connection of the channel'
                                      'signal. DC: 50 Ohm, DCLimit 1 MOhm, '
                                      'AC: Con. through DC capacitor'))

        self.add_parameter('ground',
                           label='Channel {} ground'.format(channum),
                           get_cmd='CHANnel{}:GND?'.format(channum),
                           set_cmd='CHANnel{}:GND {{}}'.format(channum),
                           vals=vals.Enum('ON', 'OFF'),
                           docstring=('Connects/disconnects the signal to/from'
                                      'the ground.'))

        # NB (WilliamHPNielsen): This parameter depends on other parameters and
        # should be dynamically updated accordingly. Cf. p 1178 of the manual
        self.add_parameter('scale',
                           label='Channel {} Y scale'.format(channum),
                           unit='V/div',
                           get_cmd='CHANnel{}:SCALe?'.format(channum),
                           set_cmd=self._set_scale,
                           get_parser=float,
                           )

        self.add_parameter('range',
                           label='Channel {} Y range'.format(channum),
                           unit='V',
                           get_cmd='CHANnel{}:RANGe?'.format(channum),
                           set_cmd=self._set_range,
                           get_parser=float
                           )

        # TODO (WilliamHPNielsen): would it be better to recast this in terms
        # of Volts?
        self.add_parameter('position',
                           label='Channel {} vert. pos.'.format(channum),
                           unit='div',
                           get_cmd='CHANnel{}:POSition?'.format(channum),
                           set_cmd='CHANnel{}:POSition {{}}'.format(channum),
                           get_parser=float,
                           vals=vals.Numbers(-5, 5),
                           docstring=('Positive values move the waveform up,'
                                      ' negative values move it down.'))

        self.add_parameter('offset',
                           label='Channel {} offset'.format(channum),
                           unit='V',
                           get_cmd='CHANnel{}:OFFSet?'.format(channum),
                           set_cmd='CHANnel{}:OFFSet {{}}'.format(channum),
                           get_parser=float,
                           )

        self.add_parameter('invert',
                           label='Channel {} inverted'.format(channum),
                           get_cmd='CHANnel{}:INVert?'.format(channum),
                           set_cmd='CHANnel{}:INVert {{}}'.format(channum),
                           vals=vals.Enum('ON', 'OFF'))

        # TODO (WilliamHPNielsen): This parameter should be dynamically
        # validated since 800 MHz BW is only available for 50 Ohm coupling
        self.add_parameter('bandwidth',
                           label='Channel {} bandwidth'.format(channum),
                           get_cmd='CHANnel{}:BANDwidth?'.format(channum),
                           set_cmd='CHANnel{}:BANDwidth {{}}'.format(channum),
                           vals=vals.Enum('FULL', 'B800', 'B200', 'B20')
                           )

        self.add_parameter('impedance',
                           label='Channel {} impedance'.format(channum),
                           unit='Ohm',
                           get_cmd='CHANnel{}:IMPedance?'.format(channum),
                           set_cmd='CHANnel{}:IMPedance {{}}'.format(channum),
                           vals=vals.Ints(1, 100000),
                           docstring=('Sets the impedance of the channel '
                                      'for power calculations and '
                                      'measurements.'))

        self.add_parameter('overload',
                           label='Channel {} overload'.format(channum),
                           get_cmd='CHANnel{}:OVERload?'.format(channum)
                           )

    #########################
    # Specialised/interlinked set/getters
    def _set_range(self, value):
        self.scale._save_val(value/10)

        self._instrument.write('CHANnel{}:RANGe {{}}'.format(self.channum,
                                                             value))

    def _set_scale(self, value):
        self.range._save_val(value*10)

        self._instrument.write('CHANnel{}:SCALe {{}}'.format(self.channum,
                                                             value))


class RTO1000(VisaInstrument):
    """
    Alpha Version of Instrument driver for the
    Rohde-Schwarz RTO1000 series oscilloscopes.

    """

    def __init__(self, name, address=None, model=None, timeout=5,
                 terminator='\n',
                 **kwargs):
        super().__init__(name=name, address=address, timeout=timeout,
                         terminator=terminator, **kwargs)

        # With firmware versions earlier than 3.65, it seems that the
        # model number can NOT be queried from the instrument
        # (at least fails with RTO1024, fw 2.52.1.1), so in that case
        # the user must provide the model manually
        firmware_version = self.get_idn()['firmware']

        if LooseVersion(firmware_version) >= LooseVersion('3.65'):
            # strip just in case there is a newline character at the end
            self.model = self.ask('DIAGnostic:SERVice:WFAModel?').strip()
            if model is not None and model != self.model:
                warnings.warn("The model number provided by the user "
                              "does not match the instrument's response."
                              " I am going to assume that this oscilloscope "
                              "is a model {}".format(self.model))
        else:
            if model is None:
                raise ValueError('No model number provided. Please provide '
                                 'a model number (eg. "RTO1024").')
            else:
                self.model = model

        # Now assign model-specific values
        self.num_chans = int(self.model[-1])

        self._horisontal_divs = int(self.ask('TIMebase:DIVisions?'))

        self.add_parameter('display',
                           label='Display state',
                           set_cmd='SYSTem:DISPlay:UPDate {}',
                           val_mapping={'remote': 0,
                                        'view': 1})

        #########################
        # Triggering

        self.add_parameter('trigger_display',
                           label='Trigger display state',
                           set_cmd='DISPlay:TRIGger:LINes {}',
                           get_cmd='DISPlay:TRIGger:LINes?',
                           val_mapping={'ON': 1, 'OFF': 0})

        # TODO: (WilliamHPNielsen) There are more available trigger
        # settings than implemented here. See p. 1261 of the manual
        # here we just use trigger1, which is the A-trigger

        self.add_parameter('trigger_source',
                           label='Trigger source',
                           set_cmd='TRIGger1:SOURce {}',
                           get_cmd='TRIGger1:SOURce?',
                           val_mapping={'CH1': 'CHAN1',
                                        'CH2': 'CHAN2',
                                        'CH3': 'CHAN3',
                                        'CH4': 'CHAN4',
                                        'EXT': 'EXT'})

        self.add_parameter('trigger_type',
                           label='Trigger type',
                           set_cmd='TRIGger1:TYPE {}',
                           get_cmd='TRIGger1:TYPE?',
                           val_mapping={'EDGE': 'EDGE',
                                        'GLITCH': 'GLIT',
                                        'WIDTH': 'WIDT',
                                        'RUNT': 'RUNT',
                                        'WINDOW': 'WIND',
                                        'TIMEOUT': 'TIM',
                                        'INTERVAL': 'INT',
                                        'SLEWRATE': 'SLEW',
                                        'DATATOCLOCK': 'DAT',
                                        'STATE': 'STAT',
                                        'PATTERN': 'PATT',
                                        'ANEDGE': 'ANED',
                                        'SERPATTERN': 'SERP',
                                        'NFC': 'NFC',
                                        'TV': 'TV',
                                        'CDR': 'CDR'}
                           )
        # See manual p. 1262 for an explanation of trigger types

        self.add_parameter('trigger_level',
                           label='Trigger level',
                           set_cmd=self._set_trigger_level,
                           get_cmd=self._get_trigger_level)

        self.add_parameter('trigger_edge_slope',
                           label='Edge trigger slope',
                           set_cmd='TRIGger1:EDGE:SLOPe {}',
                           get_cmd='TRIGger1:EDGE:SLOPe?',
                           vals=vals.Enum('POS', 'NEG', 'EITH'))

        #########################
        # Horizontal settings

        self.add_parameter('timebase_scale',
                           label='Timebase scale',
                           set_cmd=self._set_timebase_scale,
                           get_cmd='TIMebase:SCALe?',
                           unit='s/div',
                           get_parser=float,
                           vals=vals.Numbers(25e-12, 10000))

        self.add_parameter('timebase_range',
                           label='Timebase range',
                           set_cmd=self._set_timebase_range,
                           get_cmd='TIMebase:RANGe?',
                           unit='s',
                           get_parser=float,
                           vals=vals.Numbers(250e-12, 100e3))

        self.add_parameter('timebase_position',
                           label='Horizontal position',
                           set_cmd='TIMEbase:HORizontal:POSition {}',
                           get_cmd='TIMEbase:HORizontal:POSition?',
                           get_parser=float,
                           unit='s',
                           vals=vals.Numbers(-100e24, 100e24))

        #########################
        #

        self.add_parameter('num_averages',
                           label='Number of trace averages',
                           docstring='Number of averages for measuring '
                           'trace.',
                           get_cmd='ACQuire:COUNt' + '?',
                           set_cmd='ACQuire:COUNt ' + '{:.4f}',
                           vals=vals.Ints(1, 16777215),
                           get_parser=int)

        self.add_parameter('sampling_rate',
                           label='Sample rate',
                           docstring='Number of averages for measuring '
                           'trace.',
                           unit='Sa/s',
                           get_cmd='ACQuire:POINts:ARATe' + '?',
                           get_parser=int)

        # resolution does not have to equal the timescale
        # TODO : Explore timebase commands
        self.add_parameter('resolution',
                           label='Temporal resolution',
                           docstring='Resolution in seconds.',
                           unit='s',
                           get_cmd='ACQuire:RESolution' + '?',
                           set_cmd='ACQuire:RESolution ' + '{:.2f}',
                           vals=vals.Numbers(1E-15, 0.5),
                           get_parser=float)

        self.add_parameter('acq_rate',
                           label='Acquisition rate',
                           unit='Sa/s',
                           docstring='recorded waveform samples per second',
                           get_cmd='ACQuire:SRATe'+'?',
                           set_cmd='ACQuire:SRATe ' + ' {:.2f}',
                           vals=vals.Numbers(2, 20e12),
                           get_parser=float)

        self.add_parameter('t_start',
                           label='Waveform start time',
                           unit='s',
                           docstring='start time, relative to trigger in s',
                           get_cmd='EXPort:WAVeform:STARt' + '?',
                           set_cmd='EXPort:WAVeform:STARt' + ' {:.2f}',
                           vals=vals.Numbers(-100E+24, 100E+24),
                           get_parser=float)

        self.add_parameter('t_stop',
                           unit='s',
                           docstring='stop time of saved waveform'
                           'relative to trigger in s',
                           get_cmd='EXPort:WAVeform:STOP' + '?',
                           set_cmd='EXPort:WAVeform:STOP' + ' {:.2f}',
                           vals=vals.Numbers(-100E+24, 100E+24),
                           get_parser=float)

        # Add the channels to the instrument

        for ch in range(1, self.num_chans+1):
            chan = ScopeChannel(self, 'channel{}'.format(ch), ch)
            self.add_submodule('ch{}'.format(ch), chan)

        self.add_function('reset', call_cmd='*RST')
        self.add_function('opc', call_cmd='*OPC?')
        self.add_function('stop_opc', call_cmd='*STOP;OPC?')
        # starts the shutdown of the system
        self.add_function('system_shutdown', call_cmd='SYSTem:EXIT')

        self.connect_message()

    #########################
    # Specialised set/get functions

    def _set_timebase_range(self, value):
        """
        Set the full range of the timebase
        """
        self.timebase_scale._save_val(value/self._horisontal_divs)

        self.write('TIMebase:RANGe {}'.format(value))

    def _set_timebase_scale(self, value):
        """
        Set the length of one horizontal division
        """
        self.timebase_range._save_val(value*self._horisontal_divs)

        self.write('TIMebase:SCALe {}'.format(value))

    def _set_trigger_level(self, value):
        """
        Set the trigger level on the currently used trigger source
        channel
        """
        trans = {'CH1': 1, 'CH2': 2, 'CH3': 3, 'CH4': 4, 'EXT': 5}
        # we use get and not get_latest because we don't trust users to
        # not touch the front panel of an oscilloscope
        source = trans[self.trigger_source.get()]
        if source != 5:
            v_range = self.submodules['ch{}'.format(source)].range()
            offset = self.submodules['ch{}'.format(source)].offset()

            if (value < -v_range/2 + offset) or (value > v_range/2 + offset):
                raise ValueError('Trigger level outside channel range.')

        self.write('TRIGger1:LEVel{} {}'.format(source, value))

    def _get_trigger_level(self):
        """
        Get the trigger level from the currently used trigger source
        """
        trans = {'CH1': 1, 'CH2': 2, 'CH3': 3, 'CH4': 4, 'EXT': 5}
        # we use get and not get_latest because we don't trust users to
        # not touch the front panel of an oscilloscope
        source = trans[self.trigger_source.get()]

        val = self.ask('TRIGger1:LEVel{}?'.format(source))

        return float(val.strip())

    # functions below need to be edited in order to make them
    # compatible with the set inputs of the functions above
    # acquisition rate is already set
    def set_waveform(self):
        # takes as an input starting and stopping times relative to trigger
        # takes as an input no. of averages
        # this does the measurement
        # data is extracted only after this finishes
        # put a sleep command here at the end
        self.visa_handle.ask('STOP;*OPC?')
        self.visa_handle.write('EXPort:WAVeform:FASTexport ON')
        self.visa_handle.write('EXPort:WAVeform:INCXvalues ON')
        self.visa_handle.write(
            'CHANnel%s:WAVeform1:STATe 1'.format(str(self.signal_ch)))

    def get_waveform(self):
        """
        Takes as an input channel no.
        type(channel_number) = int, values from 1..4
        Method to output waveform data, as x,y values map to timestamps and voltage

        """
        # TODO: currently works for one waveform per channel,
        # the instrument supports 3 waveforms per channel
        # TODO, in Beta version: Derieve the waiting command
        # from an available visa command, replace this hacky way

        # TODO, in Beta Version: how does changing trigger interval affect acquisition time
        # TODO, in Beta Version: Play with the trigger intervals

        self.visa_handle.ask('STOP;*OPC?')
        self.visa_handle.write('RUNSingle')
        # Wait until measurement finishes before extracting data.

        sleep(self.num_averages() * self.trigger_interval() + 1)
        self.visa_handle.write(
            'EXPort:WAVeform:SOURce C%sW1'.format(str(self.signal_ch)))
        self.visa_handle.write(
            'CHANnel%s:ARIThmetics AVERage'.fomat(str(self.signal_ch)))
        ret_str = self.visa_handle.ask(
            'CHANNEL%s:WAVEFORM1:DATA?'.format(str(self.signal_ch)))
        array = ret_str.split(',')
        array = np.double(array)
        x_values = array[::2]
        y_values = array[1::2]

        return x_values, y_values
