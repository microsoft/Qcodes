from qcodes import validators as validator
from functools import partial

from .SD_Module import *


class SD_AWG(SD_Module):
    """
    This is the general SD_AWG driver class that implements shared parameters and functionality among all PXIe-based
    AWG cards by Keysight. (series M32xxA and M33xxA)

    This driver was written to be inherited from by a specific AWG card driver (e.g. M3201A).

    This driver was written with the M3201A card in mind.

    This driver makes use of the Python library provided by Keysight as part of the SD1 Software package (v.2.01.00).
    """

    def __init__(self, name, chassis, slot, channels, triggers, **kwargs):
        super().__init__(name, chassis, slot, **kwargs)

        # Create instance of keysight SD_AOU class
        self.awg = keysightSD1.SD_AOU()

        # Create an instance of keysight SD_Wave class
        self.wave = keysightSD1.SD_Wave()

        # store card-specifics
        self.channels = channels
        self.triggers = triggers

        # Open the device, using the specified chassis and slot number
        awg_name = self.awg.getProductNameBySlot(chassis, slot)
        if isinstance(awg_name, str):
            result_code = self.awg.openWithSlot(awg_name, chassis, slot)
            if result_code <= 0:
                raise Exception('Could not open SD_AWG '
                                'error code {}'.format(result_code))
        else:
            raise Exception('No SD_AWG found at '
                            'chassis {}, slot {}'.format(chassis, slot))

        self.add_parameter('trigger_io',
                           label='trigger io',
                           get_cmd=self.get_trigger_io,
                           set_cmd=self.set_trigger_io,
                           docstring='The trigger input value, 0 (OFF) or 1 (ON)',
                           vals=validator.Enum(0, 1))
        self.add_parameter('clock_frequency',
                           label='clock frequency',
                           unit='Hz',
                           get_cmd=self.get_clock_frequency,
                           set_cmd=self.set_clock_frequency,
                           docstring='The real hardware clock frequency in Hz',
                           vals=validator.Numbers(100e6, 500e6))
        self.add_parameter('clock_sync_frequency',
                           label='clock sync frequency',
                           unit='Hz',
                           get_cmd=self.get_clock_sync_frequency,
                           docstring='The frequency of the internal CLKsync in Hz')

        for i in range(triggers):
            self.add_parameter(f'pxi_trigger_number_{i}',
                               label=f'pxi trigger number {i}',
                               get_cmd=partial(self.get_pxi_trigger, pxi_trigger=(4000 + i)),
                               set_cmd=partial(self.set_pxi_trigger, pxi_trigger=(4000 + i)),
                               docstring=f'The digital value of pxi trigger no. {i}, 0 (ON) of 1 (OFF)',
                               vals=validator.Enum(0, 1))

        for i in range(channels):
            self.add_parameter(f'frequency_channel_{i}',
                               label=f'frequency channel {i}',
                               unit='Hz',
                               set_cmd=partial(self.set_channel_frequency, channel_number=i),
                               docstring=f'The frequency of channel {i}',
                               vals=validator.Numbers(0, 200e6))
            self.add_parameter(f'phase_channel_{i}',
                               label=f'phase channel {i}',
                               unit='deg',
                               set_cmd=partial(self.set_channel_phase, channel_number=i),
                               docstring=f'The phase of channel {i}',
                               vals=validator.Numbers(0, 360))
            # TODO: validate the setting of amplitude and offset at the same time (-1.5<amp+offset<1.5)
            self.add_parameter(f'amplitude_channel_{i}',
                               label=f'amplitude channel {i}',
                               unit='V',
                               set_cmd=partial(self.set_channel_amplitude, channel_number=i),
                               docstring=f'The amplitude of channel {i}',
                               vals=validator.Numbers(-1.5, 1.5))
            self.add_parameter(f'offset_channel_{i}',
                               label=f'offset channel {i}',
                               unit='V',
                               set_cmd=partial(self.set_channel_offset, channel_number=i),
                               docstring=f'The DC offset of channel {i}',
                               vals=validator.Numbers(-1.5, 1.5))
            self.add_parameter(f'wave_shape_channel_{i}',
                               label=f'wave shape channel {i}',
                               set_cmd=partial(self.set_channel_wave_shape, channel_number=i),
                               docstring=f'The output waveform type of channel {i}',
                               vals=validator.Enum(-1, 0, 1, 2, 4, 5, 6, 8))

    #
    # Get-commands
    #

    def get_trigger_io(self, verbose=False):
        """
        Reads and returns the trigger input

        Returns:
            int: Trigger input value, 0 (OFF) or 1 (ON),
            or negative numbers for errors
        """
        value = self.awg.triggerIOread()
        value_name = 'trigger_io'
        return result_parser(value, value_name, verbose)

    def get_clock_frequency(self, verbose=False):
        """
        Returns the real hardware clock frequency (CLKsys)

        Returns:
            int: real hardware clock frequency in Hz,
            or negative numbers for errors
        """
        value = self.awg.clockGetFrequency()
        value_name = 'clock_frequency'
        return result_parser(value, value_name, verbose)

    def get_clock_sync_frequency(self, verbose=False):
        """
        Returns the frequency of the internal CLKsync

        Returns:
            int: frequency of the internal CLKsync in Hz,
            or negative numbers for errors
        """
        value = self.awg.clockGetSyncFrequency()
        value_name = 'clock_sync_frequency'
        return result_parser(value, value_name, verbose)

    #
    # Set-commands
    #

    def set_clock_frequency(self, frequency, verbose=False):
        """
        Sets the module clock frequency

        Args:
            frequency (float): the frequency in Hz

        Returns:
            float: the real frequency applied to the hardware in Hw,
            or negative numbers for errors
        """
        set_frequency = self.awg.clockSetFrequency(frequency)
        value_name = 'set_clock_frequency'
        return result_parser(set_frequency, value_name, verbose)

    def set_channel_frequency(self, frequency, channel_number, verbose=False):
        """
        Sets the frequency for the specified channel.
        The frequency is used for the periodic signals generated by the Function Generators.

        Args:
            channel_number (int): output channel number
            frequency (int): frequency in Hz
        """
        value = self.awg.channelFrequency(channel_number, frequency)
        value_name = f'set frequency channel {channel_number} to {frequency} Hz'
        return result_parser(value, value_name, verbose)

    def set_channel_phase(self, phase, channel_number, verbose=False):
        """
        Sets the phase for the specified channel.

        Args:
            channel_number (int): output channel number
            phase (int): phase in degrees
        """
        value = self.awg.channelPhase(channel_number, phase)
        value_name = f'set phase channel {channel_number} to {phase} degrees'
        return result_parser(value, value_name, verbose)

    def set_channel_amplitude(self, amplitude, channel_number, verbose=False):
        """
        Sets the amplitude for the specified channel.

        Args:
            channel_number (int): output channel number
            amplitude (int): amplitude in Volts
        """
        value = self.awg.channelAmplitude(channel_number, amplitude)
        value_name = f'set amplitude channel {channel_number} to {amplitude} V'
        return result_parser(value, value_name, verbose)

    def set_channel_offset(self, offset, channel_number, verbose=False):
        """
        Sets the DC offset for the specified channel.

        Args:
            channel_number (int): output channel number
            offset (int): DC offset in Volts
        """
        value = self.awg.channelOffset(channel_number, offset)
        value_name = f'set offset channel {channel_number} to {offset} V'
        return result_parser(value, value_name, verbose)

    def set_channel_wave_shape(self, wave_shape, channel_number, verbose=False):
        """
        Sets output waveform type for the specified channel.
            HiZ         :  -1 (only available for M3202A)
            No Signal   :   0
            Sinusoidal  :   1
            Triangular  :   2
            Square      :   4
            DC Voltage  :   5
            Arbitrary wf:   6
            Partner Ch. :   8

        Args:
            channel_number (int): output channel number
            wave_shape (int): wave shape type
        """
        value = self.awg.channelWaveShape(channel_number, wave_shape)
        value_name = f'set wave shape channel {channel_number} to {wave_shape}'
        return result_parser(value, value_name, verbose)

    def set_trigger_io(self, value, verbose=False):
        """
        Sets the trigger output. The trigger must be configured as output using
        config_trigger_io

        Args:
            value (int): Tigger output value: 0 (OFF), 1 (ON)
        """
        result = self.awg.triggerIOwrite(value)
        value_name = f'set io trigger output to {value}'
        return result_parser(result, value_name, verbose)

    #
    # The methods below are useful for controlling the device, but are not used for setting or getting parameters
    #

    def off(self):
        """
        Stops the AWGs and sets the waveform of all channels to 'No Signal'
        """

        for i in range(self.channels):
            awg_response = self.awg.AWGstop(i)
            if isinstance(awg_response, int) and awg_response < 0:
                raise Exception('Error in call to Signadyne AWG '
                                'error code {}'.format(awg_response))
            channel_response = self.awg.channelWaveShape(i, 0)
            if isinstance(channel_response, int) and channel_response < 0:
                raise Exception('Error in call to Signadyne AWG '
                                'error code {}'.format(channel_response))

    def reset_clock_phase(self, trigger_behaviour, trigger_source, skew=0.0, verbose=False):
        """
        Sets the module in a sync state, waiting for the first trigger to reset the phase
        of the internal clocks CLKsync and CLKsys

        Args:
            trigger_behaviour (int): value indicating the trigger behaviour
                Active High     :   1
                Active Low      :   2
                Rising Edge     :   3
                Falling Edge    :   4

            trigger_source (int): value indicating external trigger source
                External I/O Trigger    :   0
                PXI Trigger [0..n]      :   4000+n

            skew (float) : the skew between PXI_CLK10 and CLKsync in multiples of 10ns
        """
        value = self.awg.clockResetPhase(trigger_behaviour, trigger_source, skew)
        value_name = 'reset_clock_phase trigger_behaviour: {}, trigger_source: {}, skew: {}'.format(
            trigger_behaviour, trigger_source, skew)
        return result_parser(value, value_name, verbose)

    def reset_channel_phase(self, channel_number, verbose=False):
        """
        Resets the accumulated phase of the selected channel. This accumulated phase is the
        result of the phase continuous operation of the product.

        Args:
            channel_number (int): the number of the channel to reset
        """
        value = self.awg.channelPhaseReset(channel_number)
        value_name = f'reset phase of channel {channel_number}'
        return result_parser(value, value_name, verbose)

    def reset_multiple_channel_phase(self, channel_mask, verbose=False):
        """
        Resets the accumulated phase of the selected channels simultaneously.

        Args:
            channel_mask (int): Mask to select the channel to reset (LSB is channel 0, bit 1 is channel 1 etc.)

        Example:
            reset_multiple_channel_phase(5) would reset the phase of channel 0 and 2
        """
        value = self.awg.channelPhaseResetMultiple(channel_mask)
        value_name = f'reset phase with channel mask {channel_mask}'
        return result_parser(value, value_name, verbose)

    def config_angle_modulation(self, channel_number, modulation_type, deviation_gain, verbose=False):
        """
        Configures the modulation in frequency/phase for the selected channel

        Args:
            channel_number (int): the number of the channel to configure
            modulation_type (int): the modulation type the AWG is used for
                No Modulation           :   0
                Frequency Modulation    :   1
                Phase Modulation        :   2
            deviation_gain (int): gain for the modulating signal
        """
        value = self.awg.modulationAngleConfig(channel_number, modulation_type, deviation_gain)
        value_name = 'configure angle modulation of channel {} modulation_type: {}, deviation_gain: {}'\
            .format(channel_number, modulation_type, deviation_gain)
        return result_parser(value, value_name, verbose)

    def config_amplitude_modulation(self, channel_number, modulation_type, deviation_gain, verbose=False):
        """
        Configures the modulation in amplitude/offset for the selected channel

        Args:
            channel_number (int): the number of the channel to configure
            modulation_type (int): the modulation type the AWG is used for
                No Modulation           :   0
                Amplitude Modulation    :   1
                Offset Modulation       :   2
            deviation_gain (int): gain for the modulating signal
        """
        value = self.awg.modulationAmplitudeConfig(channel_number, modulation_type, deviation_gain)
        value_name = 'configure amplitude modulation of channel {} modulation_type: {}, deviation_gain: {}' \
            .format(channel_number, modulation_type, deviation_gain)
        return result_parser(value, value_name, verbose)

    def set_iq_modulation(self, channel_number, enable, verbose=False):
        """
        Sets the IQ modulation for the selected channel

        Args:
            channel_number (int): the number of the channel to configure
            enable (int): Enable (1) or Disable (0) the IQ modulation
        """
        value = self.awg.modulationIQconfig(channel_number, enable)
        status = 'Enabled (1)' if enable == 1 else 'Disabled (0)'
        value_name = f'set IQ modulation for channel {channel_number} to {status}'
        return result_parser(value, value_name, verbose)

    def config_clock_io(self, clock_config, verbose=False):
        """
        Configures the operation of the clock output connector (CLK)

        Args:
            clock_config (int): clock connector function
                Disable         :   0   (The CLK connector is disabled)
                CLKref Output   :   1   (A copy of the reference clock is available at the CLK connector)
        """
        value = self.awg.clockIOconfig(clock_config)
        status = 'CLKref Output (1)' if clock_config == 1 else 'Disabled (0)'
        value_name = f'configure clock output connector to {status}'
        return result_parser(value, value_name, verbose)

    def config_trigger_io(self, direction, sync_mode, verbose=False):
        """
        Configures the trigger connector/line direction and synchronization/sampling method

        Args:
            direction (int): input (1) or output (0)
            sync_mode (int): sampling/synchronization mode
                Non-synchronized mode   :   0   (trigger is sampled with internal 100 Mhz clock)
                Synchronized mode       :   1   (trigger is sampled using CLK10)
        """
        value = self.awg.triggerIOconfig(direction, sync_mode)
        status = 'input (1)' if direction == 1 else 'output (0)'
        value_name = f'configure trigger io port to direction: {status}, sync_mode: {sync_mode}'
        return result_parser(value, value_name, verbose)

    #
    # Waveform related functions
    #

    def load_waveform(self, waveform_object, waveform_number, verbose=False):
        """
        Loads the specified waveform into the module onboard RAM.
        Waveforms must be created first as an instance of the SD_Wave class.

        Args:
            waveform_object (SD_Wave): pointer to the waveform object
            waveform_number (int): waveform number to identify the waveform
                in subsequent related function calls.

        Returns:
            int: available onboard RAM in waveform points,
            or negative numbers for errors
        """
        value = self.awg.waveformLoad(waveform_object, waveform_number)
        value_name = 'Loaded waveform. available_RAM'
        return result_parser(value, value_name, verbose)

    def load_waveform_int16(self, waveform_type, data_raw, waveform_number, verbose=False):
        """
        Loads the specified waveform into the module onboard RAM.
        Waveforms must be created first as an instance of the SD_Wave class.

        Args:
            waveform_type (int): waveform type
            data_raw (array): array with waveform points
            waveform_number (int): waveform number to identify the waveform
                in subsequent related function calls.

        Returns:
            int: available onboard RAM in waveform points,
            or negative numbers for errors
        """
        value = self.awg.waveformLoadInt16(waveform_type, data_raw, waveform_number)
        value_name = 'Loaded waveform. available_RAM'
        return result_parser(value, value_name, verbose)

    def reload_waveform(self, waveform_object, waveform_number, padding_mode=0, verbose=False):
        """
        Replaces a waveform located in the module onboard RAM.
        The size of the new waveform must be smaller than or
        equal to the existing waveform.

        Args:
            waveform_object (SD_Wave): pointer to the waveform object
            waveform_number (int): waveform number to identify the waveform
                in subsequent related function calls.
            padding_mode (int):
                0:  the waveform is loaded as it is, zeros are added at the
                    end if the number of points is not a multiple of the number
                    required by the AWG.
                1:  the waveform is loaded n times (using DMA) until the total
                    number of points is multiple of the number required by the
                    AWG. (only works for waveforms with even number of points)

        Returns:
            int: available onboard RAM in waveform points,
            or negative numbers for errors
        """
        value = self.awg.waveformReLoad(waveform_object, waveform_number, padding_mode)
        value_name = 'Reloaded waveform. available_RAM'
        return result_parser(value, value_name, verbose)

    def reload_waveform_int16(self, waveform_type, data_raw, waveform_number, padding_mode=0, verbose=False):
        """
        Replaces a waveform located in the module onboard RAM.
        The size of the new waveform must be smaller than or
        equal to the existing waveform.

        Args:
            waveform_type (int): waveform type
            data_raw (array): array with waveform points
            waveform_number (int): waveform number to identify the waveform
                in subsequent related function calls.
            padding_mode (int):
                0:  the waveform is loaded as it is, zeros are added at the
                    end if the number of points is not a multiple of the number
                    required by the AWG.
                1:  the waveform is loaded n times (using DMA) until the total
                    number of points is multiple of the number required by the
                    AWG. (only works for waveforms with even number of points)

        Returns:
            int: available onboard RAM in waveform points,
            or negative numbers for errors
        """
        value = self.awg.waveformReLoadArrayInt16(waveform_type, data_raw, waveform_number, padding_mode)
        value_name = 'Reloaded waveform. available_RAM'
        return result_parser(value, value_name, verbose)

    def flush_waveform(self, verbose=False):
        """
        Deletes all waveforms from the module onboard RAM and flushes all the AWG queues.
        """
        value = self.awg.waveformFlush()
        value_name = 'flushed AWG queue and RAM'
        return result_parser(value, value_name, verbose)

    #
    # AWG related functions
    #

    def awg_from_file(self, awg_number, waveform_file, trigger_mode, start_delay, cycles, prescaler, padding_mode=0,
                      verbose=False):
        """
        Provides a one-step method to load, queue and start a single waveform
        in one of the module AWGs.

        Loads a waveform from file.

        Args:
            awg_number (int): awg number where the waveform is queued
            waveform_file (str): file containing the waveform points
            trigger_mode (int): trigger method to launch the waveform
                Auto                        :   0
                Software/HVI                :   1
                Software/HVI (per cycle)    :   5
                External trigger            :   2
                External trigger (per cycle):   6
            start_delay (int): defines the delay between trigger and wf launch
                given in multiples of 10ns.
            cycles (int): number of times the waveform is repeated once launched
                zero = infinite repeats
            prescaler (int): waveform prescaler value, to reduce eff. sampling rate

        Returns:
            int: available onboard RAM in waveform points,
            or negative numbers for errors
        """
        value = self.awg.AWGFromFile(awg_number, waveform_file, trigger_mode, start_delay, cycles, prescaler,
                                     padding_mode)
        value_name = 'AWG from file. available_RAM'
        return result_parser(value, value_name, verbose)

    def awg_from_array(self, awg_number, trigger_mode, start_delay, cycles, prescaler, waveform_type, waveform_data_a,
                       waveform_data_b=None, padding_mode=0, verbose=False):
        """
        Provides a one-step method to load, queue and start a single waveform
        in one of the module AWGs.

        Loads a waveform from array.

        Args:
            awg_number (int): awg number where the waveform is queued
            trigger_mode (int): trigger method to launch the waveform
                Auto                        :   0
                Software/HVI                :   1
                Software/HVI (per cycle)    :   5
                External trigger            :   2
                External trigger (per cycle):   6
            start_delay (int): defines the delay between trigger and wf launch
                given in multiples of 10ns.
            cycles (int): number of times the waveform is repeated once launched
                zero = infinite repeats
            prescaler (int): waveform prescaler value, to reduce eff. sampling rate
            waveform_type (int): waveform type
            waveform_data_a (array): array with waveform points
            waveform_data_b (array): array with waveform points, only for the waveforms
                                        which have a second component

        Returns:
            int: available onboard RAM in waveform points,
            or negative numbers for errors
        """
        value = self.awg.AWGfromArray(awg_number, trigger_mode, start_delay, cycles, prescaler,
                                      waveform_type, waveform_data_a, waveform_data_b, padding_mode)
        value_name = 'AWG from file. available_RAM'
        return result_parser(value, value_name, verbose)

    def awg_queue_waveform(self, awg_number, waveform_number, trigger_mode, start_delay, cycles, prescaler):
        """
        Queues the specified waveform in one of the AWGs of the module.
        The waveform must be already loaded in the module onboard RAM.
        """
        self.awg.AWGqueueWaveform(awg_number, waveform_number, trigger_mode, start_delay, cycles, prescaler)

    def awg_queue_config(self, awg_number, mode):
        """
        Configures the cyclic mode of the queue. All waveforms must be already queued in one of the AWGs

        Args:
            awg_number (int): awg number where the waveform is queued
            mode (int): operation mode of the queue: One Shot (0), Cyclic (1)
        """
        self.awg.AWGqueueConfig(awg_number, mode)

    def awg_flush(self, awg_number):
        """
        Empties the queue of the selected AWG.
        Waveforms are not removed from the onboard RAM.
        """
        self.awg.AWGflush(awg_number)

    def awg_start(self, awg_number):
        """
        Starts the selected AWG from the beginning of its queue.
        The generation will start immediately or when a trigger is received,
        depending on the trigger selection of the first waveform in the queue
        and provided that at least one waveform is queued in the AWG.
        """
        self.awg.AWGstart(awg_number)

    def awg_start_multiple(self, awg_mask):
        """
        Starts the selected AWGs from the beginning of their queues.
        The generation will start immediately or when a trigger is received,
        depending on the trigger selection of the first waveform in their queues
        and provided that at least one waveform is queued in these AWGs.

        Args:
            awg_mask (int): Mask to select the awgs to start (LSB is awg 0, bit 1 is awg 1 etc.)
        """
        self.awg.AWGstartMultiple(awg_mask)

    def awg_pause(self, awg_number):
        """
        Pauses the selected AWG, leaving the last waveform point at the output,
        and ignoring all incoming triggers.
        The waveform generation can be resumed calling awg_resume
        """
        self.awg.AWGpause(awg_number)

    def awg_pause_multiple(self, awg_mask):
        """
        Pauses the selected AWGs, leaving the last waveform point at the output,
        and ignoring all incoming triggers.
        The waveform generation can be resumed calling awg_resume_multiple

        Args:
            awg_mask (int): Mask to select the awgs to pause (LSB is awg 0, bit 1 is awg 1 etc.)
        """
        self.awg.AWGpauseMultiple(awg_mask)

    def awg_resume(self, awg_number):
        """
        Resumes the selected AWG, from the current position of the queue.
        """
        self.awg.AWGresume(awg_number)

    def awg_resume_multiple(self, awg_mask):
        """
        Resumes the selected AWGs, from the current positions of their respective queue.

        Args:
            awg_mask (int): Mask to select the awgs to resume (LSB is awg 0, bit 1 is awg 1 etc.)
        """
        self.awg.AWGresumeMultiple(awg_mask)

    def awg_stop(self, awg_number):
        """
        Stops the selected AWG, setting the output to zero and resetting the AWG queue to its initial position.
        All following incoming triggers are ignored.
        """
        self.awg.AWGstop(awg_number)

    def awg_stop_multiple(self, awg_mask):
        """
        Stops the selected AWGs, setting their output to zero and resetting their AWG queues to the initial positions.
        All following incoming triggers are ignored.

        Args:
            awg_mask (int): Mask to select the awgs to stop (LSB is awg 0, bit 1 is awg 1 etc.)
        """
        self.awg.AWGstopMultiple(awg_mask)

    def awg_jump_next_waveform(self, awg_number):
        """
        Forces a jump to the next waveform in the awg queue.
        The jump is executed once the current waveform has finished a complete cycle.
        """
        self.awg.AWGjumpNextWaveform(awg_number)

    def awg_config_external_trigger(self, awg_number, external_source, trigger_behaviour):
        """
        Configures the external triggers for the selected awg.
        The external trigger is used in case the waveform is queued with th external trigger mode option.

        Args:
            awg_number (int): awg number
            external_source (int): value indicating external trigger source
                External I/O Trigger    :   0
                PXI Trigger [0..n]      :   4000+n
            trigger_behaviour (int): value indicating the trigger behaviour
                Active High     :   1
                Active Low      :   2
                Rising Edge     :   3
                Falling Edge    :   4
        """
        self.awg.AWGtriggerExternalConfig(awg_number, external_source, trigger_behaviour)

    def awg_trigger(self, awg_number):
        """
        Triggers the selected AWG.
        The waveform waiting in the current position of the queue is launched,
        provided it is configured with VI/HVI Trigger.
        """
        self.awg.AWGtrigger(awg_number)

    def awg_trigger_multiple(self, awg_mask):
        """
        Triggers the selected AWGs.
        The waveform waiting in the current position of the queue is launched,
        provided it is configured with VI/HVI Trigger.

        Args:
            awg_mask (int): Mask to select the awgs to be triggered (LSB is awg 0, bit 1 is awg 1 etc.)
        """
        self.awg.AWGtriggerMultiple(awg_mask)

    #
    # Functions related to creation of SD_Wave objects
    #

    @staticmethod
    def new_waveform_from_file(waveform_file):
        """
        Creates a SD_Wave object from data points contained in a file.
        This waveform object is stored in the PC RAM, not in the onboard RAM.

        Args:
            waveform_file (str): file containing the waveform points

        Returns:
            SD_Wave: pointer to the waveform object, or negative numbers
            for errors
        """
        wave = keysightSD1.SD_Wave()
        result = wave.newFromFile(waveform_file)
        result_parser(result)
        return wave

    @staticmethod
    def new_waveform_from_double(waveform_type, waveform_data_a, waveform_data_b=None):
        """
        Creates a SD_Wave object from data points contained in an array.
        This waveform object is stored in the PC RAM, not in the onboard RAM.

        Args:
            waveform_type (int): waveform type
            waveform_data_a (array): array of (float) with waveform points
            waveform_data_b (array): array of (float) with waveform points,
                                     only for the waveforms which have a second component

        Returns:
            SD_Wave: pointer to the waveform object, or negative numbers for
            errors
        """
        wave = keysightSD1.SD_Wave()
        result = wave.newFromArrayDouble(waveform_type, waveform_data_a, waveform_data_b)
        result_parser(result)
        return wave

    @staticmethod
    def new_waveform_from_int(waveform_type, waveform_data_a, waveform_data_b=None):
        """
        Creates a SD_Wave object from data points contained in an array.
        This waveform object is stored in the PC RAM, not in the onboard RAM.

        Args:
            waveform_type (int): waveform type
            waveform_data_a (array): array of (int) with waveform points
            waveform_data_b (array): array of (int) with waveform points,
                                     only for the waveforms which have a second component

        Returns:
            SD_Wave: pointer to the waveform object, or negative numbers
            for errors
        """
        wave = keysightSD1.SD_Wave()
        result = wave.newFromArrayInteger(waveform_type, waveform_data_a, waveform_data_b)
        result_parser(result)
        return wave

    @staticmethod
    def get_waveform_status(waveform, verbose=False):
        value = waveform.getStatus()
        value_name = 'waveform_status'
        return result_parser(value, value_name, verbose)

    @staticmethod
    def get_waveform_type(waveform, verbose=False):
        value = waveform.getType()
        value_name = 'waveform_type'
        return result_parser(value, value_name, verbose)
