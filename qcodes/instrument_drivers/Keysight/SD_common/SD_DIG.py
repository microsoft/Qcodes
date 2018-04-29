from qcodes.instrument.parameter import Parameter
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils.validators import Numbers, Enum, Ints, Anything
from scipy.interpolate import interp1d

from .SD_Module import *


# Functions to log method calls from the SD_AIN class
import re, sys, types
def logmethod(value):
    def method_wrapper(self, *args, **kwargs):
        input_str = ', '.join(map(str, args))
        if args and kwargs:
            input_str += ', ' + ', '.join(
                [f'{key}={val}' for key, val in kwargs.items()])
        method_str = f'{value.__name__}({input_str})'
        if not hasattr(self, '_method_calls'):
            self._method_calls = []
        self._method_calls += [method_str]
        return value(self, *args, **kwargs)

    return method_wrapper


def logclass(cls):
    namesToCheck = cls.__dict__.keys()

    for name in namesToCheck:
        # unbound methods show up as mere functions in the values of
        # cls.__dict__,so we have to go through getattr
        value = getattr(cls, name)
        if isinstance(value, types.FunctionType):
            setattr(cls, name, logmethod(value))
    return cls


model_channels = {'M3300A': 8}


class SignadyneParameter(Parameter):
    def __init__(self, name, channel, get_cmd=None, get_function=None,
                 set_cmd=False, set_function=None, set_args=None,
                 vals=None, val_mapping=None, initial_value=None, **kwargs):
        self.channel = channel

        self.get_cmd = get_cmd
        self.get_function = get_function

        self.set_cmd = set_cmd
        self.set_function = set_function
        self.set_args = set_args

        if val_mapping is not None:
            vals = list(val_mapping)

        super().__init__(name=name, vals=vals, val_mapping=val_mapping, **kwargs)

        if initial_value is not None:
            # We set the initial value here to ensure that it does not call
            # the set_raw method the first time
            if self.val_mapping is not None:
                initial_value = self.val_mapping[initial_value]
            self.raw_value = initial_value

    def error_check(self, value):
        assert isinstance(value, ndarray) \
               or isinstance(value, str) \
               or isinstance(value, bool) \
               or (int(value) >= 0), f'Error in call to SD_Module error code {value}'

    def set_raw(self, val):
        if self.set_cmd is not False:
            return self.set_cmd(val)
        else:
            if self.set_args is None:
                set_vals = [val]
            else:
                # Convert set args, which are parameter names, to their
                # corresponding parameter values
                set_vals = []
                for set_arg in self.set_args:
                    if set_arg == self.name:
                        set_vals.append(val)
                    else:
                        # Get the current value of the parameter
                        set_vals.append(getattr(self.channel, set_arg).raw_value)

            # Evaluate the set function with the necessary set parameter values
            return_val = self.set_function(self.channel.id, *set_vals)
            # Check if the returned value is an error
            self.error_check(return_val)

    def get_raw(self):
        if self.get_cmd is not None:
            return self.get_cmd()
        else:
            return self.get_function(self.channel.id)


class DigitizerChannel(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str, id: int, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)

        self.SD_AIN = self._parent.SD_AIN
        self.id = id

        # For channelInputConfig
        self.add_parameter(
            'full_scale',
            unit='V',
            initial_value=1,
            vals=Numbers(self.SD_AIN.channelMinFullScale(),
                         self.SD_AIN.channelMaxFullScale()),
            get_function= self.SD_AIN.channelFullScale,
            set_function=self.SD_AIN.channelInputConfig,
            docstring=f'The full scale voltage for ch{self.id}'
        )

        # For channelTriggerConfig
        self.add_parameter(
            'impedance',
            initial_value='50',
            val_mapping={'high': 0, '50': 1},
            get_function= self.SD_AIN.channelImpedance,
            set_function=self.SD_AIN.channelInputConfig,
            set_args=['full_scale', 'impedance', 'coupling'],
            docstring=f'The input impedance of ch{self.id}. Note that for '
                      f'high input impedance, the measured voltage will not be '
                      f'the actual voltage'
        )

        self.add_parameter(
            'coupling',
            initial_value='AC',
            val_mapping={'DC': 0, 'AC': 1},
            get_function=self.SD_AIN.channelCoupling,
            set_function=self.SD_AIN.channelInputConfig,
            set_args=['full_scale', 'impedance', 'coupling'],
            docstring=f'The coupling of ch{self.id}'
        )

        # For channelPrescalerConfig
        self.add_parameter(
            'prescaler',
            initial_value=0,
            vals=Ints(0, 4095),
            get_function=self.SD_AIN.channelPrescalerConfig,
            set_function=self.SD_AIN.channelPrescalerConfig,
            docstring=f'The sampling frequency prescaler for ch{self.id}. '
                      f'Sampling rate will be max_sampling_rate/(prescaler+1)'
        )

        # For channelTriggerConfig
        self.add_parameter(
            'trigger_edge',
            initial_value='rising',
            val_mapping={'rising': 1, 'falling': 2, 'both': 3},
            set_function=self.SD_AIN.channelTriggerConfig,
            set_args=['trigger_edge', 'trigger_threshold'],
            docstring=f'The trigger mode for ch{self.id}'
        )

        self.add_parameter(
            'trigger_threshold',
            initial_value=0,
            vals=Numbers(-3, 3),
            set_function=self.SD_AIN.channelTriggerConfig,
            set_args=['trigger_edge', 'trigger_threshold'],
            docstring=f'the value in volts for the trigger threshold'
        )

        # For DAQ config
        self.add_parameter(
            'points_per_cycle',
            initial_value=0,
            vals=Ints(),
            set_function=self.SD_AIN.DAQconfig,
            set_args=['points_per_cycle', 'n_cycles', 'trigger_delay_samples', 'trigger_mode'],
            docstring=f'The number of points per cycle for ch{self.id}'
        )

        self.add_parameter(
            'n_cycles',
            initial_value=-1,
            vals=Ints(),
            set_function=self.SD_AIN.DAQconfig,
            set_args=['points_per_cycle', 'n_cycles', 'trigger_delay_samples', 'trigger_mode'],
            docstring=f'The number of cycles to collect on DAQ {self.id}'
        )

        self.add_parameter(
            'trigger_delay_samples',
            initial_value=0,
            vals=Ints(),
            set_function=self.SD_AIN.DAQconfig,
            set_args=['points_per_cycle', 'n_cycles', 'trigger_delay_samples', 'trigger_mode'],
            docstring=f'The trigger delay for DAQ {self.id}. Can be negative'
        )

        self.add_parameter(
            'trigger_mode',
            initial_value='auto',
            val_mapping={'auto': 0, 'software': 1, 'digital': 2, 'analog': 3},
            set_function=self.SD_AIN.DAQconfig,
            set_args=['points_per_cycle', 'n_cycles', 'trigger_delay_samples', 'trigger_mode'],
            docstring=f'The trigger mode for ch{self.id}'
        )

        # For DAQ trigger Config
        self.add_parameter(
            'digital_trigger_mode',
            initial_value='rising_edge',
            val_mapping={'active_high': 1, 'active_low': 2,
                         'rising_edge': 3, 'falling_edge': 4},
            set_function=self.SD_AIN.DAQdigitalTriggerConfig,
            set_args=['digital_trigger_source', 'digital_trigger_mode'],
            docstring='The digital trigger mode. Can be `active_high`, '
                      '`active_low`, `rising_edge`, `falling_edge`'
        )

        self.add_parameter(
            'digital_trigger_source',
            initial_value='trig_in',
            val_mapping={'trig_in': 0, **{f'pxi{k}': 4000+k for k in range(1, 9)}},
            set_function=self.SD_AIN.DAQdigitalTriggerConfig,
            docstring='the trigger source you are using. Can be trig_in '
                      '(external IO) or pxi1 to pxi8'
        )

        self.add_parameter(
            'analog_trigger_mask',
            initial_value=0,
            vals=Ints(),
            set_function=self.SD_AIN.DAQanalogTriggerConfig,
            docstring='the trigger mask you are using. Each bit signifies '
                      'which analog channel to trigger on. The channel trigger'
                      ' behaviour must be configured separately.'
        )

        # For DAQ read
        self.add_parameter(
            'n_points',
            initial_value=0,
            vals=Ints(),
            set_cmd=None,
            docstring='the number of points to be read from specified DAQ'
        )

        self.add_parameter(
            'timeout',
            unit='s',
            initial_value=-1,
            vals=Numbers(min_value=0),
            set_cmd=None,
            docstring=f'The read timeout in seconds. 0 means infinite.'
                      f'Warning: setting to 0 will freeze the digitizer until'
                      f'acquisition has completed.'
        )


class SD_DIG(SD_Module):
    """
    This is the qcodes driver for a generic Keysight Digitizer of the M32/33XX series.

    This driver is written with the M3300A in mind.

    This driver makes use of the Python library provided by Keysight as part of the SD1 Software package (v.2.01.00).
    """
    
    def __init__(self, name, model, chassis, slot, channels=None, triggers=8, **kwargs):
        """ Initialises a generic Signadyne digitizer and its parameters

            Args:
                name (str)      : the name of the digitizer card
                channels (int)  : the number of input channels the specified card has
                triggers (int)  : the number of trigger inputs the specified card has
        """
        super().__init__(name, model, chassis, slot, triggers, **kwargs)


        if channels is None:
            channels = model_channels[self.model]

        # Create instance of keysight SD_AIN class
        # We wrap it in a logclass so that any method call is recorded in
        # self.SD_AIN._method_calls
        self.SD_AIN = logclass(keysightSD1.SD_AIN)()

        # store card-specifics
        self.n_channels = channels

        # Open the device, using the specified chassis and slot number
        dig_name = self.SD_AIN.getProductNameBySlot(chassis, slot)
        if isinstance(dig_name, str):
            result_code = self.SD_AIN.openWithSlot(dig_name, chassis, slot)
            if result_code <= 0:
                raise Exception('Could not open SD_DIG '
                                'error code {}'.format(result_code))
        else:
            raise Exception('No SD_DIG found at '
                            'chassis {}, slot {}'.format(chassis, slot))

        # for triggerIOconfig
        self.add_parameter(
            'trigger_direction',
            label='Trigger direction for trigger port',
            vals=Enum(0, 1),
            set_cmd=self.SD_AIN.triggerIOconfig,
            docstring='The trigger direction for digitizer trigger port'
        )

        # for clockSetFrequency
        self.add_parameter(
            'system_frequency',
            label='System clock frequency',
            vals=Ints(),
            set_cmd=self.SD_AIN.clockSetFrequency,
            get_cmd=self.SD_AIN.clockGetFrequency,
            docstring='The frequency of internal CLKsys in Hz'
        )

        # for clockGetSyncFrequency
        self.add_parameter(
            'sync_frequency',
            label='Clock synchronization frequency',
            vals=Ints(),
            get_cmd=self.SD_AIN.clockGetSyncFrequency,
            docstring='The frequency of internal CLKsync in Hz'
        )

        self.add_parameter('trigger_io',
                           label='trigger io',
                           get_cmd=self.get_trigger_io,
                           set_cmd=self.set_trigger_io,
                           docstring='The trigger input value, 0 (OFF) or 1 (ON)',
                           vals=Enum(0, 1))

        self.channels = ChannelList(self,
                                    name='channels',
                                    chan_type=DigitizerChannel)
        for ch in range(self.n_channels):
            channel = DigitizerChannel(self, name=f'ch{ch}', id=ch)
            setattr(self, f'ch{ch}', channel)
            self.channels.append(channel)

    def daq_read(self, daq, verbose=False):
        """ Read from the specified DAQ

        Args:
            daq (int)       : the input DAQ you are reading from

        Parameters:
            n_points
            timeout
        """
        value = self.SD_AIN.DAQread(daq, self._n_points[daq], int(self._timeout[daq]*1e3))
        if not isinstance(value, int):
            # Scale signal from int to volts, why are we checking for non-int?
            int_min, int_max = -0x8000, 0x7FFF
            v_min, v_max = -self._full_scale[daq], self._full_scale[daq]
            relative_value = (value.astype(float) - int_min) / (int_max - int_min)
            scaled_value = v_min + (v_max-v_min) * relative_value
        else:
            scaled_value = value
        value_name = f'DAQ_read channel {daq}'
        return result_parser(scaled_value, value_name, verbose)

    def daq_start(self, daq):
        """ Start acquiring data or waiting for a trigger on the specified DAQ

        Args:
            daq (int)       : the input DAQ you are enabling
        """
        return result_parser(value=self.SD_AIN.DAQstart(daq),
                             name=f'DAQ_start channel {daq}')

    def daq_start_multiple(self, daq_mask, verbose=False):
        """ Start acquiring data or waiting for a trigger on the specified DAQs

        Args:
            daq_mask (int)  : the input DAQs you are enabling, composed as a bitmask
                              where the LSB is for DAQ_0, bit 1 is for DAQ_1 etc.
        """
        value = self.SD_AIN.DAQstartMultiple(daq_mask)
        value_name = 'DAQ_start_multiple mask {:#b}'.format(daq_mask)
        return result_parser(value, value_name, verbose)

    def daq_stop(self, daq, verbose=False):
        """ Stop acquiring data on the specified DAQ

        Args:
            daq (int)       : the DAQ you are disabling
        """
        value = self.SD_AIN.DAQstop(daq)
        value_name = 'DAQ_stop channel {}'.format(daq)
        return result_parser(value, value_name, verbose)

    def daq_stop_multiple(self, daq_mask, verbose=False):
        """ Stop acquiring data on the specified DAQs

        Args:
            daq_mask (int)  : the DAQs you are triggering, composed as a bitmask
                              where the LSB is for DAQ_0, bit 1 is for DAQ_1 etc.
        """
        value = self.SD_AIN.DAQstopMultiple(daq_mask)
        value_name = 'DAQ_stop_multiple mask {:#b}'.format(daq_mask)
        return result_parser(value, value_name, verbose)

    def daq_trigger(self, daq, verbose=False):
        """ Manually trigger the specified DAQ

        Args:
            daq (int)       : the DAQ you are triggering
        """
        value = self.SD_AIN.DAQtrigger(daq)
        value_name = 'DAQ_trigger channel {}'.format(daq)
        return result_parser(value, value_name, verbose)

    def daq_trigger_multiple(self, daq_mask, verbose=False):
        """ Manually trigger the specified DAQs

        Args:
            daq_mask (int)  : the DAQs you are triggering, composed as a bitmask
                              where the LSB is for DAQ_0, bit 1 is for DAQ_1 etc.
        """
        value = self.SD_AIN.DAQtriggerMultiple(daq_mask)
        value_name = 'DAQ_trigger_multiple mask {:#b}'.format(daq_mask)
        return result_parser(value, value_name, verbose)

    def daq_flush(self, daq, verbose=False):
        """ Flush the specified DAQ

        Args:
            daq (int)       : the DAQ you are flushing
        """
        value = self.SD_AIN.DAQflush(daq)
        value_name = 'DAQ_flush channel {}'.format(daq)
        return result_parser(value, value_name, verbose)

    def daq_flush_multiple(self, daq_mask, verbose=False):
        """ Flush the specified DAQs

        Args:
            daq_mask (int)  : the DAQs you are flushing, composed as a bitmask
                              where the LSB is for DAQ_0, bit 1 is for DAQ_1 etc.
        """
        value = self.SD_AIN.DAQflushMultiple(daq_mask)
        value_name = 'DAQ_flush_multiple mask {:#b}'.format(daq_mask)
        return result_parser(value, value_name, verbose)

    def set_trigger_io(self, val, verbose=False):
        """ Write a value to the IO trigger port

        Args:
            value (int)     : the binary value to write to the IO port

        """
        # TODO: Check if the port is writable
        value = self.SD_AIN.triggerIOwrite(val)
        value_name = 'set io trigger output to {}'.format(val)
        return result_parser(value, value_name, verbose)

    def get_trigger_io(self, verbose=False):
        """ Write a value to the IO trigger port

        """
        # TODO: Check if the port is readable
        value = self.SD_AIN.triggerIOread()
        value_name = 'trigger_io'
        return result_parser(value, value_name, verbose)

    def reset_clock_phase(self, trigger_behaviour, trigger_source, skew=0.0, verbose=False):
        """ Reset the clock phase between CLKsync and CLKsys

        Args:
            trigger_behaviour (int) :
            trigger_source    (int) : the PXI trigger number
            skew           (double) : the skew between PXI_CLK10 and CLKsync in multiples of 10ns

        """
        value = self.SD_AIN.clockResetPhase(trigger_behaviour, trigger_source, skew)
        value_name = 'reset_clock_phase trigger_behaviour: {}, trigger_source: {}, skew: {}'.format(
            trigger_behaviour, trigger_source, skew)
        return result_parser(value, value_name, verbose)

    @staticmethod
    def set_clksys_frequency(frequency, verbose=False):
        """ Sets the CLKsys frequency

        Args:

        frequency (int)         : frequency of CLKsys in Hz

        """
        value = 0
        value_name = 'set_CLKsys_frequency not implemented'
        return result_parser(value, value_name, verbose)
