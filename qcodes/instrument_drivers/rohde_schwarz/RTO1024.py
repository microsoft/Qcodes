
import numpy as np
from time import sleep
from qcodes.instrument import visa
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter


class RTO1024_scope_test(visa.VisaInstrument):
    '''
    Alpha Version of Instrument driver for the 
    Rohde-Schwarz RTO1024 oscilloscope.
    Currently only able to measure signal on Channel 1.
    Contains commands for acquiring an average waveform.

    '''

    def __init__(self, name, address=None, timeout=5, terminator='',
                 **kwargs):
        super().__init__(name=name, address=address, timeout=timeout,
                         terminator=terminator, **kwargs)

        self.add_parameter('num_averages',
                           docstring='Number of averages for measuring '
                           'trace.',
                           get_cmd='ACQuire:COUNt' + '?',
                           set_cmd='ACQuire:COUNt ' + '{:.4f}',
                           vals=vals.Ints(1, 16777215),
                           get_parser=int)
        self.add_parameter('Sampling_rate',
                           docstring='Number of averages for measuring '
                           'trace.',
                           units='Sa/s',
                           get_cmd='ACQuire:POINts:ARATe' + '?',
                           get_parser=int)
        # resolution doesnot have to be the timescale
        # TODO : Explore timebase commands
        self.add_parameter('time_step',
                           docstring='Resolution in seconds.',
                           units='s',
                           get_cmd='ACQuire:RESolution' + '?',
                           set_cmd='ACQuire:RESolution ' + '{:.2f}',
                           vals=vals.Numbers(1E-15, 0.5),
                           get_parser=float)

        # some difficulty in finding a command(s) for this from the manual
        # replace this with an appropriate command(s) from the manual
        self.add_parameter('trigger_interval',
                           docstring='Time between triggers',
                           vals=vals.Numbers(),
                           units='s',
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
                           units='Sa/s',
                           docstring='recorded waveform samples per second',
                           get_cmd='ACQuire:SRATe'+'?',
                           set_cmd='ACQuire:SRATe ' + ' {:.2f}',
                           vals=vals.Numbers(2, 20e12),
                           get_parser=float)
        # needs get set methods as channel no. needs to be satisfied
        self.add_parameter('y_range',
                           units='Volt/div',
                           get_cmd='CHANnel1:RANGe'+'?',
                           set_cmd=self.set_y_range,
                           docstring='voltage range 10 vertical divisions',
                           # todo:something fishy  here
                           vals=vals.Numbers(.01, 10),
                           get_parser=float)
        self.add_parameter('y_scale',
                           units='Volt/div',
                           get_cmd='CHANnel1:SCALe'+'?',
                           set_cmd=self.set_y_scale,
                           docstring='Scale value',
                           vals=vals.Numbers(.001, 1),
                           get_parser=float)

        self.add_parameter('t_start',
                           units='s',
                           docstring='start time, relative to trigger in s',
                           get_cmd='EXPort:WAVeform:STARt' + '?',
                           set_cmd='EXPort:WAVeform:STARt' + ' {:.2f}',
                           vals=vals.Numbers(-100E+24, 100E+24),
                           get_parser=float)
        self.add_parameter('t_stop',
                           units='s',
                           docstring='stop ime of saved waveform'
                           'relative to trigger in s',
                           get_cmd='EXPort:WAVeform:STOP' + '?',
                           set_cmd='EXPort:WAVeform:STOP' + ' {:.2f}',
                           vals=vals.Numbers(-100E+24, 100E+24),
                           get_parser=float)

        self.add_function('reset', call_cmd='*RST')
        self.add_function('opc', call_cmd='*OPC?')
        self.add_function('stop_opc', call_cmd='*STOP;OPC?')
        # starts the shutdown of the system
        self.add_function('system_shutdown', call_cmd='SYSTem:EXIT')
        self.initialise()

    # test these functions

    def get_y_range(self):
        self.visa_handle.ask('CHANnel%s:RANGe?' % str(self.signal_ch))

    def set_y_range(self, val):
        # get the float here correctly
        self.visa_handle.write(
            'CHANnel%s:RANGe{:.4f}'.format(str(self.signal_ch), val))

    def get_y_scale(self):
        self.visa_handle.ask('CHANnel%s:SCALe?' % str(self.signal_ch))

    def set_y_scale(self, val):
        self.visa_handle.write(
            'CHANnel%s:SCALe{:.4f}'.format(str(self.signal_ch), val))

    def initialise(self):
        # sets the time_step, y_range, trigger and channel source
        # also sets t_start, t_stop and num_averages

        # this function is for quick debugging, will be removed in the beta
        # version

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
