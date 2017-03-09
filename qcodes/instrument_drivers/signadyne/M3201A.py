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
                           docstring='The trigger input value, 0 (OFF) or 1 (ON)')
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
                               docstring='The digital value of pxi trigger no. {}, 0 (ON) of 1 (OFF)'.format(i))

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

    #
    # The methods below are useful for controlling the device, but are not used for setting or getting parameters
    #

    # closes the hardware device and also throws away the current instrument object
    # if you want to open the instrument again, you have to initialize a new instrument object
    def close(self):
        self.awg.close()
        super().close()

    # only closes the hardware device, not the current instrument object
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
