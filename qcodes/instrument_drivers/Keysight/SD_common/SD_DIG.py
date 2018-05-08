from typing import Callable, List
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils import validators as vals

from .SD_Module import SD_Module, keysightSD1, SignadyneParameter, with_error_check


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


class DigitizerChannel(InstrumentChannel):
    """Signadyne digitizer channel

    Args:
        parent: Parent Signadyne digitizer Instrument
        name: channel name (e.g. 'ch1')
        id: channel id (e.g. 1)
        **kwargs: Additional kwargs passed to InstrumentChannel
    """
    def __init__(self, parent: Instrument, name: str, id: int, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)

        self.SD_AIN = self._parent.SD_AIN
        self.id = id

        # For channelInputConfig
        self.add_parameter(
            'full_scale',
            unit='V',
            initial_value=1,
            vals=vals.Numbers(0, 3),
            # self.SD_AIN.channelMinFullScale(),
            #                   self.SD_AIN.channelMaxFullScale()),
            set_function=self.SD_AIN.channelInputConfig,
            set_args=['full_scale', 'impedance', 'coupling'],
            docstring=f'The full scale voltage for ch{self.id}'
        )

        # For channelTriggerConfig
        self.add_parameter(
            'impedance',
            initial_value='50',
            val_mapping={'high': 0, '50': 1},
            get_function=self.SD_AIN.channelImpedance,
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
            vals=vals.Ints(0, 4095),
            get_function=self.SD_AIN.channelPrescalerConfig,
            set_function=self.SD_AIN.channelPrescalerConfig,
            docstring=f'The sampling frequency prescaler for ch{self.id}. '
                      f'Sampling rate will be max_sampling_rate/(prescaler+1)'
        )

        # For DAQ config
        self.add_parameter(
            'points_per_cycle',
            initial_value=0,
            vals=vals.Ints(),
            set_function=self.SD_AIN.DAQconfig,
            set_args=['points_per_cycle', 'n_cycles',
                      'trigger_delay_samples', 'trigger_mode'],
            docstring=f'The number of points per cycle for ch{self.id}'
        )

        self.add_parameter(
            'n_cycles',
            initial_value=-1,
            vals=vals.Ints(),
            set_function=self.SD_AIN.DAQconfig,
            set_args=['points_per_cycle', 'n_cycles',
                      'trigger_delay_samples', 'trigger_mode'],
            docstring=f'The number of cycles to collect on DAQ {self.id}'
        )

        self.add_parameter(
            'trigger_mode',
            initial_value='auto',
            val_mapping={'auto': 0, 'software': 1, 'digital': 2, 'analog': 3},
            set_function=self.SD_AIN.DAQconfig,
            set_args=['points_per_cycle', 'n_cycles',
                      'trigger_delay_samples', 'trigger_mode'],
            docstring=f'The trigger mode for ch{self.id}'
        )

        self.add_parameter(
            'trigger_delay_samples',
            initial_value=0,
            vals=vals.Numbers(),
            set_parser=int,
            set_function=self.SD_AIN.DAQconfig,
            set_args=['points_per_cycle', 'n_cycles',
                      'trigger_delay_samples', 'trigger_mode'],
            docstring=f'The trigger delay (in samples) for ch{self.id}. '
                      f'Can be negative'
        )

        # For channelTriggerConfig
        self.add_parameter(
            'analog_trigger_edge',
            initial_value='rising',
            val_mapping={'rising': 1, 'falling': 2, 'both': 3},
            set_function=self.SD_AIN.channelTriggerConfig,
            set_args=['analog_trigger_edge', 'analog_trigger_threshold'],
            docstring=f'The analog trigger edge for ch{self.id}.'
                      f'This is only used when the channel is set as the analog'
                      f'trigger channel'
        )

        self.add_parameter(
            'analog_trigger_threshold',
            initial_value=0,
            vals=vals.Numbers(-3, 3),
            set_function=self.SD_AIN.channelTriggerConfig,
            set_args=['analog_trigger_edge', 'analog_trigger_threshold'],
            docstring=f'the value in volts for the trigger threshold'
        )

        self.add_parameter(
            'analog_trigger_mask',
            initial_value=0,
            vals=vals.Ints(),
            set_function=self.SD_AIN.DAQanalogTriggerConfig,
            docstring='the trigger mask you are using. Each bit signifies '
                      'which analog channel to trigger on. The channel trigger'
                      ' behaviour must be configured separately (trigger_edge '
                      'and trigger_threshold). Needs to be double checked, but '
                      'it seems multiple analog trigger channels can be used.'
        )

        # For DAQ trigger Config
        self.add_parameter(
            'digital_trigger_mode',
            initial_value='rising',
            val_mapping={'active_high': 1, 'active_low': 2,
                         'rising': 3, 'falling': 4},
            set_function=self.SD_AIN.DAQdigitalTriggerConfig,
            set_args=['digital_trigger_source', 'digital_trigger_mode'],
            docstring='The digital trigger mode. Can be `active_high`, '
                      '`active_low`, `rising`, `falling`'
        )

        self.add_parameter(
            'digital_trigger_source',
            initial_value='trig_in',
            val_mapping={'trig_in': 0, **{f'pxi{k}': 4000+k for k in range(8)}},
            set_function=self.SD_AIN.DAQdigitalTriggerConfig,
            set_args=['digital_trigger_source', 'digital_trigger_mode'],
            docstring='the trigger source you are using. Can be trig_in '
                      '(external IO) or pxi0 to pxi7'
        )

        # For DAQ read
        self.add_parameter(
            'n_points',
            initial_value=0,
            vals=vals.Ints(),
            set_cmd=None,
            docstring='the number of points to be read from specified DAQ'
        )

        self.add_parameter(
            'timeout',
            unit='s',
            initial_value=-1,
            vals=vals.Numbers(min_value=0),
            set_cmd=None,
            docstring=f'The read timeout in seconds. 0 means infinite.'
                      f'Warning: setting to 0 will freeze the digitizer until'
                      f'acquisition has completed.'
        )

    def add_parameter(self, name: str,
                      parameter_class: type=SignadyneParameter, **kwargs):
        """Use SignadyneParameter by default"""
        super().add_parameter(name=name, parameter_class=parameter_class,
                              parent=self, **kwargs)

    @with_error_check
    def start(self):
        """ Start acquiring data or waiting for a trigger on the specified DAQ

        Acquisition data can then be read using `daq_read`

        Raises:
            AssertionError if DAQstart was unsuccessful
        """
        return self.SD_AIN.DAQstart(self.id)

    @with_error_check
    def read(self) -> np.ndarray:
        """ Read from the specified DAQ.

        Channel acquisition must first be started using `daq_start`
        Uses channel parameters `n_points` and `timeout`

        Returns:
            Numpy array with acquisition data

        Raises:
            AssertionError if DAQread was unsuccessful
        """
        value = self.SD_AIN.DAQread(self.id, self.n_points(),
                                    int(self.timeout() * 1e3))  # ms
        if not isinstance(value, int):
            # Scale signal from int to volts, why are we checking for non-int?
            int_min, int_max = -0x8000, 0x7FFF
            v_min, v_max = -self.full_scale(), self.full_scale()
            relative_value = (value.astype(float) - int_min) / (int_max - int_min)
            scaled_value = v_min + (v_max-v_min) * relative_value
        else:
            scaled_value = value
        return scaled_value

    @with_error_check
    def stop(self):
        """ Stop acquiring data on the specified DAQ

        Raises:
            AssertionError if DAQstop was unsuccessful
        """
        return self.SD_AIN.DAQstop(self.id)

    @with_error_check
    def flush(self):
        """ Flush the DAQ channel

        Raises:
            AssertionError if DAQflush was unsuccessful
        """
        return self.SD_AIN.DAQflush(self.id)

    @with_error_check
    def trigger(self):
        """ Manually trigger the specified DAQ

        Raises:
            AssertionError if DAQtrigger was unsuccessful
        """
        return self.SD_AIN.DAQtrigger(self.id)



class SD_DIG(SD_Module):
    """Qcodes driver for a generic Keysight Digitizer of the M32/33XX series.

    This driver is written with the M3300A in mind.

    This driver makes use of the Python library provided by Keysight as part of
    the SD1 Software package (v.2.01.00).

    Args:
        name: the name of the digitizer card
        model: Digitizer model (e.g. 'M3300A').
            Used to retrieve number of channels if not specified
        chassis: Signadyne chassis (usually 0).
        slot: module slot in chassis (starting at 1)
        channels: the number of input channels the specified card has
        triggers: the number of pxi trigger inputs the specified card has
    """

    def __init__(self,
                 name: str,
                 model: str,
                 chassis: int,
                 slot: int,
                 channels: int = None,
                 triggers: int = 8,
                 **kwargs):
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
        self.initialize(chassis=chassis, slot=slot)

        # for triggerIOconfig
        self.add_parameter(
            'trigger_direction',
            label='Trigger direction for trigger port',
            val_mapping={'out': 0, 'in': 1},
            set_cmd=self.SD_AIN.triggerIOconfig,
            docstring='The trigger direction for digitizer trigger port'
        )

        # for clockSetFrequency
        self.add_parameter(
            'system_frequency',
            label='System clock frequency',
            vals=vals.Numbers(),
            set_cmd=None,
            initial_value=100e6,
            # clockGetFrequency seems to give issues
            # set_cmd=self.SD_AIN.clockSetFrequency,
            # get_cmd=self.SD_AIN.clockGetFrequency,
            docstring='The frequency of internal CLKsys in Hz'
        )

        # for clockGetSyncFrequency
        self.add_parameter(
            'sync_frequency',
            label='Clock synchronization frequency',
            vals=vals.Ints(),
            get_cmd=self.SD_AIN.clockGetSyncFrequency,
            docstring='The frequency of internal CLKsync in Hz'
        )

        self.add_parameter('trigger_io',
                           label='trigger io',
                           get_function=self.SD_AIN.triggerIOread,
                           set_functon=self.SD_AIN.triggerIOwrite,
                           docstring='The trigger input value, 0 (OFF) or 1 (ON)',
                           val_mapping={'off': 0, 'on': 1})

        self.channels = ChannelList(self,
                                    name='channels',
                                    chan_type=DigitizerChannel)
        for ch in range(self.n_channels):
            channel = DigitizerChannel(self, name=f'ch{ch}', id=ch)
            setattr(self, f'ch{ch}', channel)
            self.channels.append(channel)

    def add_parameter(self, name: str,
                      parameter_class: type=SignadyneParameter, **kwargs):
        """Use SignadyneParameter by default"""
        super().add_parameter(name=name, parameter_class=parameter_class,
                              parent=self, **kwargs)

    def initialize(self, chassis: int, slot: int):
        """Open connection to digitizer

        Args:
            chassis: Signadyne chassis number (usually 1)
            slot: Module slot in chassis

        Returns:
            Name of digitizer

        Raises:
            AssertionError if connection to digitizer was unsuccessful
        """
        digitizer_name = self.SD_AIN.getProductNameBySlot(chassis, slot)
        assert isinstance(digitizer_name, str), \
            f'No SD_DIG found at chassis {chassis}, slot {slot}'

        result_code = self.SD_AIN.openWithSlot(digitizer_name, chassis, slot)
        assert result_code > 0, f'Could not open SD_DIG error code {result_code}'

        return digitizer_name

    @with_error_check
    def start_channels(self, channels: List[int]):
        """ Start acquiring data or waiting for a trigger on the specified DAQs

        Args:
            channels: list of channels to start

        Raises:
            AssertionError if DAQstartMultiple was unsuccessful

        """
        # DAQ channel mask, where LSB is for DAQ_0, bit 1 is for DAQ_1 etc.
        channel_mask = sum(2**channel for channel in channels)
        return self.SD_AIN.DAQstartMultiple(channel_mask)

    @with_error_check
    def stop_channels(self, channels: List[int]):
        """ Stop acquiring data on the specified DAQs

        Args:
            channels: List of DAQ channels to stop

        Raises:
            AssertionError if DAQstopMultiple was unsuccessful
        """
        # DAQ channel mask, where LSB is for DAQ_0, bit 1 is for DAQ_1 etc.
        channel_mask = sum(2**channel for channel in channels)
        return self.SD_AIN.DAQstopMultiple(channel_mask)

    @with_error_check
    def trigger_channels(self, channels):
        """ Manually trigger the specified DAQs

        Args:
            channels: List of DAQ channels to trigger

        Raises:
            AssertionError if DAQtriggerMultiple was unsuccessful
        """

        # DAQ channel mask, where LSB is for DAQ_0, bit 1 is for DAQ_1 etc.
        channel_mask = sum(2**channel for channel in channels)
        return self.SD_AIN.DAQtriggerMultiple(channel_mask)

    @with_error_check
    def flush_channels(self, channels: List[int]):
        """ Flush the specified DAQ channels

        Args:
            channels: List of DAQ channels to flush

        Raises:
            AssertionError if DAQflushMultiple was unsuccessful
        """
        # DAQ channel mask, where LSB is for DAQ_0, bit 1 is for DAQ_1 etc.
        channel_mask = sum(2**channel for channel in channels)
        return self.SD_AIN.DAQflushMultiple(channel_mask)

    @with_error_check
    def reset_clock_phase(self,
                          trigger_behaviour: int,
                          trigger_source: int,
                          skew: float = 0.0):
        """ Reset the clock phase between CLKsync and CLKsys

        Args:
            trigger_behaviour:
            trigger_source: the PXI trigger number
            skew: the skew between PXI_CLK10 and CLKsync in multiples of 10ns

        Raises:
            AssertionError if clockResetPhase was unsuccessful
        """
        return self.SD_AIN.clockResetPhase(trigger_behaviour, trigger_source, skew)
