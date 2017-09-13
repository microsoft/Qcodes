
import numpy as np
from time import sleep
from qcodes.instrument import visa, Instrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter


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

        # Add the parameters
        # Note (WilliamHPNielsen): I just slavishly typed in pp 1176-1180
        # of the manual
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

        # NOTE (WilliamHPNielsen): This parameter depends on other parameters and
        # should be dynamically updated accordingly. Cf. p 1178 of the manual
        self.add_parameter('scale',
                           label='Channel {} Y scale'.format(channum),
                           unit='V/div',
                           get_cmd='CHANnel{}:SCALe?'.format(channum),
                           set_cmd='CHANnel{}:SCALe {{}}'.format(channum),
                           )

        # NOTE (WilliamHPNielsen):
        # This parameter competes with the `scale` parameter, and they should
        # be interlinked
        self.add_parameter('range',
                           label='Channel {} Y range'.format(channum),
                           unit='V/div',
                           get_cmd='CHANnel{}:RANGe?'.format(channum),
                           set_cmd='CHANnel{}:RANGe {{}}'.format(channum),
                           )

        # TODO (WilliamHPNielsen): would it be better to recast this in terms
        # of Volts?
        self.add_parameter('position',
                           label='Channel {} vert. pos.'.format(channum),
                           unit='div',
                           get_cmd='CHANnel{}:POSition?'.format(channum),
                           set_cmd='CHANnel{}:POSition {{}}'.format(channum),
                           vals=vals.Numbers(-5, 5),
                           docstring=('Positive values move the waveform up,'
                                      ' negative values move it down.'))

        self.add_parameter('offset',
                           label='Channel {} offset'.format(channum),
                           unit='V',
                           get_cmd='CHANnel{}:OFFSet?'.format(channum),
                           set_cmd='CHANnel{}:OFFSet {{}}'.format(channum),
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


class RTO1024_scope_test(visa.VisaInstrument):
    """
    Alpha Version of Instrument driver for the
    Rohde-Schwarz RTO1024 oscilloscope.
    Currently only able to measure signal on Channel 1.
    Contains commands for acquiring an average waveform.

    TODO (WilliamHPNielsen):
        * Channelise the channel settings
        * Cast waveform/trace acquisition into an array parameter
    """

    def __init__(self, name, address=None, timeout=5, terminator='',
                 **kwargs):
        super().__init__(name=name, address=address, timeout=timeout,
                         terminator=terminator, **kwargs)

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

        # some difficulty in finding a command(s) for this from the manual
        # replace this with an appropriate command(s) from the manual
        self.add_parameter('trigger_interval',
                           docstring='Time between triggers',
                           vals=vals.Numbers(),
                           unit='s',
                           initial_value=1,
                           parameter_class=ManualParameter)

        # some difficulty in finding a command(s) for this from the manual
        # replace this with an appropriate command(s) from the manual
        self.add_parameter('trigger_ch',
                           docstring='Trigger channel',
                           vals=vals.Ints(1, 4),
                           initial_value=2,
                           parameter_class=ManualParameter)

        # write a get-set method for signal channel,
        # and all other methods dependent on signal channel
        self.add_parameter('signal_ch',
                           docstring='Signal channel',
                           vals=vals.Ints(1, 4),
                           initial_value=1,
                           parameter_class=ManualParameter)

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
                           units='s',
                           docstring='stop time of saved waveform'
                           'relative to trigger in s',
                           get_cmd='EXPort:WAVeform:STOP' + '?',
                           set_cmd='EXPort:WAVeform:STOP' + ' {:.2f}',
                           vals=vals.Numbers(-100E+24, 100E+24),
                           get_parser=float)

        # Add the channels to the instrument

        channels = ChannelList(self, 'channels', ScopeChannel)

        for ch in range(1, 5):
            chan = ScopeChannel(self, 'channel{}'.format(ch), ch)
            channels.append(chan)
            self.add_submodule('ch{}'.format(ch), chan)
        channels.lock()
        self.add_submodule('channels', channels)

        self.add_function('reset', call_cmd='*RST')
        self.add_function('opc', call_cmd='*OPC?')
        self.add_function('stop_opc', call_cmd='*STOP;OPC?')
        # starts the shutdown of the system
        self.add_function('system_shutdown', call_cmd='SYSTem:EXIT')
        self.initialise()

    def initialise(self):
        # sets the time_step, y_range, trigger and channel source
        # also sets t_start, t_stop and num_averages

        # this function is for quick debugging, will be removed in the beta
        # version

        # NOTE (WilliamHPNielsen): I broke this function with the
        # channelisation

        self.write('*RST')
        self.time_step(1E-8)
        self.num_averages(10)
        self.t_start(-50e-08)
        self.t_stop(50e-08)
        self.y_range(0.5)
        self.y_scale(0.05)

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
