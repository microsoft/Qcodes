from qcodes.instrument.base import Instrument
from qcodes import validators as validator
from functools import partial
try:
    import signadyne
except ImportError:
    raise ImportError('to use the M32 driver install the signadyne module')


def result_parser(value, name, verbose=False):
    """
    This method is used for parsing the result in the get-methods.
    For values that are non-negative, the value is simply returned.
    Negative values indicate an error, so an error is raised
    with a reference to the error code.

    The parser also can print to the result to the shell if verbose is 1.

    Args:
        value: the value to be parsed
        name (str): name of the value to be parsed
        verbose (bool): boolean indicating verbose mode

    Returns:
        value: parsed value, which is the same as value if non-negative
        or not a number
    """
    if isinstance(value, str) or isinstance(value, bool) or (int(value) >= 0):
        if verbose:
            print('{}: {}' .format(name, value))
        return value
    else:
        raise Exception('Error in call to Signadyne AWG '
                        'error code {}'.format(value))


class Signadyne_M3201A(Instrument):
    """
    This is the qcodes driver for the Signadyne M32/M33xx series of function/arbitrary waveform generators

    status: beta-version

    This driver is written with the M3201A in mind.
    Updates might/will be necessary for other versions of Signadyne cards.

    Args:
        name (str): name for this instrument, passed to the base instrument
        chassis (int): chassis number where the device is located
        slot (int): slot number where the device is plugged in
    """

    def __init__(self, name, chassis=1, slot=7, **kwargs):
        super().__init__(name, **kwargs)

        # Create instance of signadyne SD_AOU class
        self.awg = signadyne.SD_AOU()

        # Open the device, using the specified chassis and slot number
        awg_name = self.awg.getProductNameBySlot(chassis, slot)
        if isinstance(awg_name, str):
            result_code = self.awg.openWithSlot(awg_name, chassis, slot)
            if result_code <= 0:
                raise Exception('Could not open Signadyne AWG '
                                'error code {}'.format(result_code))
        else:
            raise Exception('Signadyne AWG not found at '
                            'chassis {}, slot {}'.format(chassis, slot))

        self.add_parameter('module_count',
                           label='module count',
                           get_cmd=self.get_module_count,
                           docstring='The number of Signadyne modules installed in the system')
        self.add_parameter('product_name',
                           label='product name',
                           get_cmd=self.get_product_name,
                           docstring='The product name of the device')
        self.add_parameter('serial_number',
                           label='serial number',
                           get_cmd=self.get_serial_number,
                           docstring='The serial number of the device')
        self.add_parameter('chassis_number',
                           label='chassis number',
                           get_cmd=self.get_chassis,
                           docstring='The chassis number where the device is located')
        self.add_parameter('slot_number',
                           label='slot number',
                           get_cmd=self.get_slot,
                           docstring='The slot number where the device is located')
        self.add_parameter('status',
                           label='status',
                           get_cmd=self.get_status,
                           docstring='The status of the device')
        self.add_parameter('firmware_version',
                           label='firmware version',
                           get_cmd=self.get_firmware_version,
                           docstring='The firmware version of the device')
        self.add_parameter('hardware_version',
                           label='hardware version',
                           get_cmd=self.get_hardware_version,
                           docstring='The hardware version of the device')
        self.add_parameter('instrument_type',
                           label='type',
                           get_cmd=self.get_type,
                           docstring='The type of the device')
        self.add_parameter('open',
                           label='open',
                           get_cmd=self.get_open,
                           docstring='Indicating if device is open, True (open) or False (closed)')
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

        for i in [0, 1, 2, 3, 4, 5, 6, 7]:
            self.add_parameter('pxi_trigger_number_{}'.format(i),
                               label='pxi trigger number {}'.format(i),
                               get_cmd=partial(self.get_pxi_trigger, pxi_trigger=(4000 + i)),
                               set_cmd=partial(self.set_pxi_trigger, pxi_trigger=(4000 + i)),
                               docstring='The digital value of pxi trigger no. {}, 0 (ON) of 1 (OFF)'.format(i),
                               vals=validator.Enum(0, 1))

        for i in [0, 1, 2, 3]:
            self.add_parameter('frequency_channel_{}'.format(i),
                               label='frequency channel {}'.format(i),
                               unit='Hz',
                               set_cmd=partial(self.set_channel_frequency, channel_number=i),
                               docstring='The frequency of channel {}'.format(i),
                               vals=validator.Numbers(0, 200e6))
            self.add_parameter('phase_channel_{}'.format(i),
                               label='phase channel {}'.format(i),
                               unit='deg',
                               set_cmd=partial(self.set_channel_phase, channel_number=i),
                               docstring='The phase of channel {}'.format(i),
                               vals=validator.Numbers(0, 360))
            # TODO: validate the setting of amplitude and offset at the same time (-1.5<amp+offset<1.5)
            self.add_parameter('amplitude_channel_{}'.format(i),
                               label='amplitude channel {}'.format(i),
                               unit='V',
                               set_cmd=partial(self.set_channel_amplitude, channel_number=i),
                               docstring='The amplitude of channel {}'.format(i),
                               vals=validator.Numbers(-1.5, 1.5))
            self.add_parameter('offset_channel_{}'.format(i),
                               label='offset channel {}'.format(i),
                               unit='V',
                               set_cmd=partial(self.set_channel_offset, channel_number=i),
                               docstring='The DC offset of channel {}'.format(i),
                               vals=validator.Numbers(-1.5, 1.5))
            self.add_parameter('wave_shape_channel_{}'.format(i),
                               label='wave shape channel {}'.format(i),
                               set_cmd=partial(self.set_channel_wave_shape, channel_number=i),
                               docstring='The output waveform type of channel {}'.format(i),
                               vals=validator.Enum(-1, 1, 2, 4, 5, 6, 8))

    #
    # Get-commands
    #

    def get_module_count(self, verbose=False):
        """Returns the number of Signadyne modules installed in the system"""
        value = self.awg.moduleCount()
        value_name = 'module_count'
        return result_parser(value, value_name, verbose)

    def get_product_name(self, verbose=False):
        """Returns the product name of the device"""
        value = self.awg.getProductName()
        value_name = 'product_name'
        return result_parser(value, value_name, verbose)

    def get_serial_number(self, verbose=False):
        """Returns the serial number of the device"""
        value = self.awg.getSerialNumber()
        value_name = 'serial_number'
        return result_parser(value, value_name, verbose)

    def get_chassis(self, verbose=False):
        """Returns the chassis number where the device is located"""
        value = self.awg.getChassis()
        value_name = 'chassis_number'
        return result_parser(value, value_name, verbose)

    def get_slot(self, verbose=False):
        """Returns the slot number where the device is located"""
        value = self.awg.getSlot()
        value_name = 'slot_number'
        return result_parser(value, value_name, verbose)

    def get_status(self, verbose=False):
        """Returns the status of the device"""
        value = self.awg.getStatus()
        value_name = 'status'
        return result_parser(value, value_name, verbose)

    def get_firmware_version(self, verbose=False):
        """Returns the firmware version of the device"""
        value = self.awg.getFirmwareVersion()
        value_name = 'firmware_version'
        return result_parser(value, value_name, verbose)

    def get_hardware_version(self, verbose=False):
        """Returns the hardware version of the device"""
        value = self.awg.getHardwareVersion()
        value_name = 'hardware_version'
        return result_parser(value, value_name, verbose)

    def get_type(self, verbose=False):
        """Returns the type of the device"""
        value = self.awg.getType()
        value_name = 'type'
        return result_parser(value, value_name, verbose)

    def get_open(self, verbose=False):
        """Returns whether the device is open (True) or not (False)"""
        value = self.awg.isOpen()
        value_name = 'open'
        return result_parser(value, value_name, verbose)

    def get_pxi_trigger(self, pxi_trigger, verbose=False):
        """
        Returns the digital value of the specified PXI trigger

        Args:
            pxi_trigger (int): PXI trigger number (4000 + Trigger No.)
            verbose (bool): boolean indicating verbose mode

        Returns:
            value (int): Digital value with negated logic, 0 (ON) or 1 (OFF),
            or negative numbers for errors
        """
        value = self.awg.PXItriggerRead(pxi_trigger)
        value_name = 'pxi_trigger number {}'.format(pxi_trigger)
        return result_parser(value, value_name, verbose)

    def get_trigger_io(self, verbose=False):
        """
        Reads and returns the trigger input

        Returns:
            value (int): Trigger input value, 0 (OFF) or 1 (ON),
            or negative numbers for errors
        """
        value = self.awg.triggerIOread()
        value_name = 'trigger_io'
        return result_parser(value, value_name, verbose)

    def get_clock_frequency(self, verbose=False):
        """
        Returns the real hardware clock frequency (CLKsys)

        Returns:
            value (int): real hardware clock frequency in Hz,
            or negative numbers for errors
        """
        value = self.awg.clockGetFrequency()
        value_name = 'clock_frequency'
        return result_parser(value, value_name, verbose)

    def get_clock_sync_frequency(self, verbose=False):
        """
        Returns the frequency of the internal CLKsync

        Returns:
            value (int): frequency of the internal CLKsync in Hz,
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
            set_frequency (float): the real frequency applied to the hardware in Hw,
            or negative numbers for errors
        """
        set_frequency = self.awg.clockSetFrequency(frequency)
        value_name = 'set_clock_frequency'
        return result_parser(set_frequency, value_name, verbose)

    def set_channel_frequency(self, frequency, channel_number):
        """
        Sets the frequency for the specified channel.
        The frequency is used for the periodic signals generated by the Function Generators.

        Args:
            channel_number (int): output channel number
            frequency (int): frequency in Hz
        """
        self.awg.channelFrequency(channel_number, frequency)

    def set_channel_phase(self, phase, channel_number):
        """
        Sets the phase for the specified channel.

        Args:
            channel_number (int): output channel number
            phase (int): phase in degrees
        """
        self.awg.channelPhase(channel_number, phase)

    def set_channel_amplitude(self, amplitude, channel_number):
        """
        Sets the amplitude for the specified channel.

        Args:
            channel_number (int): output channel number
            amplitude (int): amplitude in Volts
        """
        self.awg.channelAmplitude(channel_number, amplitude)

    def set_channel_offset(self, offset, channel_number):
        """
        Sets the DC offset for the specified channel.

        Args:
            channel_number (int): output channel number
            offset (int): DC offset in Volts
        """
        self.awg.channelOffset(channel_number, offset)

    def set_channel_wave_shape(self, wave_shape, channel_number):
        """
        Sets output waveform type for the specified channel.
            No Signal   :  -1
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
        self.awg.channelWaveShape(channel_number, wave_shape)

    def set_pxi_trigger(self, value, pxi_trigger):
        """
        Sets the digital value of the specified PXI trigger

        Args:
            pxi_trigger (int): PXI trigger number (4000 + Trigger No.)
            value (int): Digital value with negated logic, 0 (ON) or 1 (OFF)
        """
        self.awg.PXItriggerWrite(pxi_trigger, value)

    def set_trigger_io(self, value):
        """
        Sets the trigger output. The trigger must be configured as output using
        config_trigger_io

        Args:
            value (int): Tigger output value: 0 (OFF), 1 (ON)
        """
        self.awg.triggerIOwrite(value)

    #
    # The methods below are useful for controlling the device, but are not used for setting or getting parameters
    #

    # closes the hardware device and also throws away the current instrument object
    # if you want to open the instrument again, you have to initialize a new instrument object
    def close(self):
        self.awg.close()
        super().close()

    # only closes the hardware device, does not delete the current instrument object
    def close_soft(self):
        self.awg.close()

    def off(self):
        """
        Stops the AWGs and sets the waveform of all channels to 'No Signal'
        """

        for i in [0, 1, 2, 3]:
            awg_response = self.awg.AWGstop(i)
            if isinstance(awg_response, int) and awg_response < 0:
                raise Exception('Error in call to Signadyne AWG '
                                'error code {}'.format(awg_response))
            channel_response = self.awg.channelWaveShape(i, -1)
            if isinstance(channel_response, int) and channel_response < 0:
                raise Exception('Error in call to Signadyne AWG '
                                'error code {}'.format(channel_response))

    def open_with_serial_number(self, name, serial_number):
        self.awg.openWithSerialNumber(name, serial_number)

    def open_with_slot(self, name, chassis, slot):
        self.awg.openWithSlot(name, chassis, slot)

    def run_self_test(self):
        value = self.awg.runSelfTest()
        print('Did self test and got result: {}'.format(value))

    def reset_clock_phase(self, trigger_behaviour, trigger_source):
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
        """
        self.awg.clockResetPhase(trigger_behaviour, trigger_source)

    def reset_channel_phase(self, channel_number):
        """
        Resets the accumulated phase of the selected channel. This accumulated phase is the
        result of the phase continuous operation of the product.

        Args:
            channel_number (int): the number of the channel to reset
        """
        self.awg.channelPhaseReset(channel_number)

    def reset_multiple_channel_phase(self, channel_mask):
        """
        Resets the accumulated phase of the selected channels simultaneously.

        Args:
            channel_mask (int): Mask to select the channel to reset (LSB is channel 0, bit 1 is channel 1 etc.)

        Example:
            reset_multiple_channel_phase(5) would reset the phase of channel 0 and 2
        """
        self.awg.channelPhaseResetMultiple(channel_mask)

    def config_angle_modulation(self, channel_number, modulation_type, deviation_gain):
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
        self.awg.modulationAngleConfig(channel_number, modulation_type, deviation_gain)

    def config_amplitude_modulation(self, channel_number, modulation_type, deviation_gain):
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
        self.awg.modulationAmplitudeConfig(channel_number, modulation_type, deviation_gain)

    def set_iq_modulation(self, channel_number, enable):
        """
        Sets the IQ modulation for the selected channel

        Args:
            channel_number (int): the number of the channel to configure
            enable (int): Enable (1) or Disabled (0) the IQ modulation
        """
        self.awg.modulationIQconfig(channel_number, enable)

    def config_clock_io(self, clock_config):
        """
        Sets the IQ modulation for the selected channel

        Args:
            clock_config (int): clock connector function
                Disable         :   0   (The CLK connector is disabled)
                CLKref Output   :   1   (A copy of the reference clock is available at the CLK connector)
        """
        self.awg.clockIOconfig(clock_config)

    def config_trigger_io(self, direction, sync_mode):
        """
        Configures the trigger connector/line direction and synchronization/sampling method

        Args:
            direction (int): input (1) or output (0)
            sync_mode (int): sampling/synchronization mode
                Non-synchronized mode   :   0   (trigger is sampled with internal 100 Mhz clock)
                Synchronized mode       :   1   (trigger is sampled using CLK10)
        """
        self.awg.triggerIOconfig(direction, sync_mode)

    #
    # Waveform related functions
    #

    def load_wave_form(self, waveform_object, waveform_number, verbose=False):
        """
        Loads the specified waveform into the module onboard RAM.
        Waveforms must be created first as an instance of the SD_Wave class.

        Args:
            waveform_object (SD_Wave): pointer to the waveform object
            waveform_number (int): waveform number to identify the waveform
                in subsequent related function calls.

        Returns:
            availableRAM (int): available onboard RAM in waveform points,
            or negative numbers for errors
        """
        value = self.awg.waveformLoad(waveform_object, waveform_number)
        value_name = 'Loaded waveform. available_RAM'
        return result_parser(value, value_name, verbose)

    def load_wave_form_int16(self, waveform_type, data_raw, waveform_number, verbose=False):
        """
        Loads the specified waveform into the module onboard RAM.
        Waveforms must be created first as an instance of the SD_Wave class.

        Args:
            waveform_type (int): waveform type
            data_raw (array): array with waveform points
            waveform_number (int): waveform number to identify the waveform
                in subsequent related function calls.

        Returns:
            availableRAM (int): available onboard RAM in waveform points,
            or negative numbers for errors
        """
        value = self.awg.waveformLoadInt16(waveform_type, data_raw, waveform_number)
        value_name = 'Loaded waveform. available_RAM'
        return result_parser(value, value_name, verbose)

    def reload_wave_form(self, waveform_object, waveform_number, padding_mode=0, verbose=False):
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
            availableRAM (int): available onboard RAM in waveform points,
            or negative numbers for errors
        """
        value = self.awg.waveformReLoad(waveform_object, waveform_number, padding_mode)
        value_name = 'Reloaded waveform. available_RAM'
        return result_parser(value, value_name, verbose)

    def reload_wave_form_int16(self, waveform_type, data_raw, waveform_number, padding_mode=0, verbose=False):
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
            availableRAM (int): available onboard RAM in waveform points,
            or negative numbers for errors
        """
        value = self.awg.waveformReLoadArrayInt16(waveform_type, data_raw, waveform_number, padding_mode)
        value_name = 'Reloaded waveform. available_RAM'
        return result_parser(value, value_name, verbose)

    def flush_wave_form(self, verbose=False):
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
                negative = infinite repeats
            prescaler (int): waveform prescaler value, to reduce eff. sampling rate

        Returns:
            availableRAM (int): available onboard RAM in waveform points,
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
                negative = infinite repeats
            prescaler (int): waveform prescaler value, to reduce eff. sampling rate
            waveform_type (int): waveform type
            waveform_data_a (array): array with waveform points
            waveform_data_b (array): array with waveform points, only for the waveforms
                                        which have a second component

        Returns:
            availableRAM (int): available onboard RAM in waveform points,
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

    def awg_jump_next_wave_form(self, awg_number):
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
            awg_mask (int): Mask to select the awgs to stop (LSB is awg 0, bit 1 is awg 1 etc.)
        """
        self.awg.AWGtriggerMultiple(awg_mask)

    #
    # The methods below are not used for setting or getting parameters, but can be used in the test functions of the
    # test suite e.g. The main reason they are defined is to make this driver more complete
    #

    def get_product_name_by_slot(self, chassis, slot, verbose=False):
        value = self.awg.getProductNameBySlot(chassis, slot)
        value_name = 'product_name'
        return result_parser(value, value_name, verbose)

    def get_product_name_by_index(self, index, verbose=False):
        value = self.awg.getProductNameByIndex(index)
        value_name = 'product_name'
        return result_parser(value, value_name, verbose)

    def get_serial_number_by_slot(self, chassis, slot, verbose=False):
        value = self.awg.getSerialNumberBySlot(chassis, slot)
        value_name = 'serial_number'
        return result_parser(value, value_name, verbose)

    def get_serial_number_by_index(self, index, verbose=False):
        value = self.awg.getSerialNumberByIndex(index)
        value_name = 'serial_number'
        return result_parser(value, value_name, verbose)

    # method below is commented out because it is missing from the dll provided by Signadyne
    # def get_type_by_slot(self, chassis, slot, verbose=False):
    #     value = self.awg.getTypeBySlot(chassis, slot)
    #     value_name = 'type'
    #     return result_parser(value, value_name, verbose)

    # method below is commented out because it is missing from the dll provided by Signadyne
    # def get_type_by_index(self, index, verbose=False):
    #     value = self.awg.getTypeByIndex(index)
    #     value_name = 'type'
    #     return result_parser(value, value_name, verbose)

    # method below is commented out because it is missing from the dll provided by Signadyne
    # def get_awg_running(self, verbose=0, awg_number):
    #     """
    #     Returns whether the AWG is running or stopped
    #
    #     Args:
    #         awg_number (int): AWG number
    #
    #     Returns:
    #         value (int): 1 if the AWG is running, 0 if it is stopped
    #     """
    #     value =
    #     if verbose:
    #         print('slot_number: %s' % value)
    #     return value

    # method below is commented out because it is missing from the dll provided by Signadyne
    # def get_awg_waveform_number_playing(self, verbose=0, awg_number=0):
    #     """
    #     Returns the waveformNumber of the waveform which is currently being generated.
    #
    #     Args:
    #         awg_number (int): AWG number
    #
    #     Returns:
    #         value (int): Waveform identifier,
    #         or negative numbers for errors
    #     """
    #     value = self.awg.AWG
    #     if verbose:
    #         print('pxi_trigger number %s: %s' % (pxi_trigger, value))
    #     return value
