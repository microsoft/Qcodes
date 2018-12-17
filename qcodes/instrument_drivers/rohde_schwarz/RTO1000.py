# All manual references are to R&S RTO Digital Oscilloscope User Manual
# for firmware 3.65, 2017

import logging
import warnings
import time

import numpy as np
from distutils.version import LooseVersion

from qcodes import Instrument
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ArrayParameter

log = logging.getLogger(__name__)


class ScopeTrace(ArrayParameter):

    def __init__(self, name: str, instrument: InstrumentChannel,
                 channum: int) -> None:
        """
        The ScopeTrace parameter is attached to a channel of the oscilloscope.

        For now, we only support reading out the entire trace.
        """
        super().__init__(name=name,
                         shape=(1,),
                         label='Voltage',  # TODO: Is this sometimes dbm?
                         unit='V',
                         setpoint_names=('Time',),
                         setpoint_labels=('Time',),
                         setpoint_units=('s',),
                         docstring='Holds scope trace')

        self.channel = instrument
        self.channum = channum

    def prepare_trace(self) -> None:
        """
        Prepare the scope for returning data, calculate the setpoints
        """
        # We always use 16 bit integers for the data format
        self.channel._parent.dataformat('INT,16')
        # ensure little-endianess
        self.channel._parent.write('FORMat:BORder LSBFirst')
        # only export y-values
        self.channel._parent.write('EXPort:WAVeform:INCXvalues OFF')
        # only export one channel
        self.channel._parent.write('EXPort:WAVeform:MULTichannel OFF')

        # now get setpoints

        hdr = self.channel._parent.ask('CHANnel{}:'.format(self.channum) +
                                       'DATA:HEADER?')
        hdr_vals = list(map(float, hdr.split(',')))
        t_start = hdr_vals[0]
        t_stop = hdr_vals[1]
        no_samples = int(hdr_vals[2])
        values_per_sample = hdr_vals[3]

        # NOTE (WilliamHPNielsen):
        # if samples are multi-valued, we need a MultiParameter
        # instead of an arrayparameter
        if values_per_sample > 1:
            raise NotImplementedError('There are several values per sample '
                                      'in this trace (are you using envelope'
                                      ' or peak detect?). We currently do '
                                      'not support saving such a trace.')

        self.shape = (no_samples,)
        self.setpoints = (tuple(np.linspace(t_start, t_stop, no_samples)),)

        self._trace_ready = True
        # we must ensure that all this took effect before proceeding
        self.channel._parent.ask('*OPC?')

    def get_raw(self):
        """
        Returns a trace
        """

        instr = self.channel._parent

        if not self._trace_ready:
            raise ValueError('Trace not ready! Please call '
                             'prepare_trace().')

        if instr.run_mode() == 'RUN Nx SINGLE':
            N = instr.num_acquisitions()
            M = instr.completed_acquisitions()
            log.info('Acquiring {} traces.'.format(N))
            while M < N:
                log.info('Acquired {}:{} traces.'.format(M, N))
                time.sleep(0.25)
                M = instr.completed_acquisitions()

        log.info('Acquisition completed. Polling trace from instrument.')
        vh = instr.visa_handle
        vh.write('CHANnel{}:DATA?'.format(self.channum))
        raw_vals = vh.read_raw()

        num_length = int(raw_vals[1:2])
        no_points = int(raw_vals[2:2+num_length])

        # cut of the header and the trailing '\n'
        raw_vals = raw_vals[2+num_length:-1]

        dataformat = instr.dataformat.get_latest()

        if dataformat == 'INT,8':
            int_vals = np.fromstring(raw_vals, dtype=np.int8, count=no_points)
        else:
            int_vals = np.fromstring(raw_vals, dtype=np.int16,
                                     count=no_points//2)

        # now the integer values must be converted to physical
        # values

        scale = self.channel.scale()
        no_divs = 10  # TODO: Is this ever NOT 10?

        # we always export as 16 bit integers
        quant_levels = 253*256
        conv_factor = scale*no_divs/quant_levels
        output = conv_factor*int_vals + self.channel.offset()

        return output


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

        self.add_parameter('arithmetics',
                           label='Channel {} arithmetics'.format(channum),
                           set_cmd='CHANnel{}:ARIThmetics {{}}'.format(channum),
                           get_cmd='CHANnel{}:ARIThmetics?'.format(channum),
                           val_mapping={'AVERAGE': 'AVER',
                                        'OFF': 'OFF',
                                        'ENVELOPE': 'ENV'}
                           )

        #########################
        # Trace

        self.add_parameter('trace',
                           channum=self.channum,
                           parameter_class=ScopeTrace)

        self._trace_ready = False

    #########################
    # Specialised/interlinked set/getters
    def _set_range(self, value):
        self.scale._save_val(value/10)

        self._parent.write('CHANnel{}:RANGe {}'.format(self.channum,
                                                       value))

    def _set_scale(self, value):
        self.range._save_val(value*10)

        self._parent.write('CHANnel{}:SCALe {}'.format(self.channum,
                                                       value))


class RTO1000(VisaInstrument):
    """
    QCoDeS Instrument driver for the
    Rohde-Schwarz RTO1000 series oscilloscopes.

    """

    def __init__(self, name: str, address: str,
                 model: str=None, timeout: float=5.,
                 HD: bool=True,
                 terminator: str='\n',
                 **kwargs) -> None:
        """
        Args:
            name: name of the instrument
            address: VISA resource address
            model: The instrument model. For newer firmware versions,
                this can be auto-detected
            timeout: The VISA query timeout
            HD: Does the unit have the High Definition Option (allowing
                16 bit vertical resolution)
            terminator: Command termination character to strip from VISA
                commands.
        """
        super().__init__(name=name, address=address, timeout=timeout,
                         terminator=terminator, **kwargs)

        # With firmware versions earlier than 3.65, it seems that the
        # model number can NOT be queried from the instrument
        # (at least fails with RTO1024, fw 2.52.1.1), so in that case
        # the user must provide the model manually
        firmware_version = self.get_idn()['firmware']

        if LooseVersion(firmware_version) < LooseVersion('3'):
            log.warning('Old firmware version detected. This driver may '
                        'not be compatible. Please upgrade your firmware.')

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

        self.HD = HD

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
                           set_cmd=self._set_timebase_position,
                           get_cmd='TIMEbase:HORizontal:POSition?',
                           get_parser=float,
                           unit='s',
                           vals=vals.Numbers(-100e24, 100e24))

        #########################
        # Acquisition

        # I couldn't find a way to query the run mode, so we manually keep
        # track of it. It is very important when getting the trace to make
        # sense of completed_acquisitions
        self.add_parameter('run_mode',
                           label='Run/acqusition mode of the scope',
                           get_cmd=None,
                           set_cmd=None)

        self.run_mode('RUN CONT')

        self.add_parameter('num_acquisitions',
                           label='Number of single acquisitions to perform',
                           get_cmd='ACQuire:COUNt?',
                           set_cmd='ACQuire:COUNt {}',
                           vals=vals.Ints(1, 16777215),
                           get_parser=int)

        self.add_parameter('completed_acquisitions',
                           label='Number of completed acquisitions',
                           get_cmd='ACQuire:CURRent?',
                           get_parser=int)

        self.add_parameter('sampling_rate',
                           label='Sample rate',
                           docstring='Number of averages for measuring '
                           'trace.',
                           unit='Sa/s',
                           get_cmd='ACQuire:POINts:ARATe' + '?',
                           get_parser=int)

        self.add_parameter('acquisition_sample_rate',
                           label='Acquisition sample rate',
                           unit='Sa/s',
                           docstring='recorded waveform samples per second',
                           get_cmd='ACQuire:SRATe'+'?',
                           set_cmd='ACQuire:SRATe ' + ' {:.2f}',
                           vals=vals.Numbers(2, 20e12),
                           get_parser=float)

        #########################
        # Data

        self.add_parameter('dataformat',
                           label='Export data format',
                           set_cmd='FORMat:DATA {}',
                           get_cmd='FORMat:DATA?',
                           vals=vals.Enum('ASC,0', 'REAL,32',
                                          'INT,8', 'INT,16'))

        #########################
        # High definition mode (might not be available on all instruments)

        if HD:
            self.add_parameter('high_definition_state',
                               label='High definition (16 bit) state',
                               set_cmd=self._set_hd_mode,
                               get_cmd='HDEFinition:STAte?',
                               val_mapping={'ON': 1, 'OFF': 0})

            self.add_parameter('high_definition_bandwidth',
                               label='High definition mode bandwidth',
                               set_cmd='HDEFinition:BWIDth {}',
                               get_cmd='HDEFinition:BWIDth?',
                               unit='Hz',
                               get_parser=float,
                               vals=vals.Numbers(1e4, 1e9))

        # Add the channels to the instrument
        for ch in range(1, self.num_chans+1):
            chan = ScopeChannel(self, 'channel{}'.format(ch), ch)
            self.add_submodule('ch{}'.format(ch), chan)

        self.add_function('stop', call_cmd='STOP')
        self.add_function('reset', call_cmd='*RST')
        self.add_function('opc', call_cmd='*OPC?')
        self.add_function('stop_opc', call_cmd='*STOP;OPC?')
        # starts the shutdown of the system
        self.add_function('system_shutdown', call_cmd='SYSTem:EXIT')

        self.connect_message()

    def run_cont(self) -> None:
        """
        Set the instrument in 'RUN CONT' mode
        """
        self.write('RUN')
        self.run_mode.set('RUN CONT')

    def run_single(self) -> None:
        """
        Set the instrument in 'RUN Nx SINGLE' mode
        """
        self.write('SINGLE')
        self.run_mode.set('RUN Nx SINGLE')

    #########################
    # Specialised set/get functions

    def _set_hd_mode(self, value):
        """
        Set/unset the high def mode
        """
        self._make_traces_not_ready()
        self.write('HDEFinition:STAte {}'.format(value))

    def _set_timebase_range(self, value):
        """
        Set the full range of the timebase
        """
        self._make_traces_not_ready()
        self.timebase_scale._save_val(value/self._horisontal_divs)

        self.write('TIMebase:RANGe {}'.format(value))

    def _set_timebase_scale(self, value):
        """
        Set the length of one horizontal division
        """
        self._make_traces_not_ready()
        self.timebase_range._save_val(value*self._horisontal_divs)

        self.write('TIMebase:SCALe {}'.format(value))

    def _set_timebase_position(self, value):
        """
        Set the horizontal position
        """
        self._make_traces_not_ready()
        self.write('TIMEbase:HORizontal:POSition {}'.format(value))

    def _make_traces_not_ready(self):
        """
        Make the scope traces be not ready
        """
        self.ch1.trace._trace_ready = False
        self.ch2.trace._trace_ready = False
        self.ch3.trace._trace_ready = False
        self.ch4.trace._trace_ready = False

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
