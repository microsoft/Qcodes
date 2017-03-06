from qcodes.instrument.base import Instrument
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
    if isinstance(value, str) or (int(value) >= 0):
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

    def __init__(self, name, chassis=1, slot=8, **kwargs):
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
        self.add_parameter('trigger_io',
                           label='trigger io',
                           get_cmd=self.get_trigger_io,
                           docstring='The trigger input value, 0 (OFF) or 1 (ON)')
        self.add_parameter('clock_frequency',
                           label='clock frequency',
                           unit='Hz',
                           get_cmd=self.get_clock_frequency,
                           docstring='The real hardware clock frequency in Hz')
        self.add_parameter('clock_sync_frequency',
                           label='clock sync frequency',
                           unit='Hz',
                           get_cmd=self.get_clock_sync_frequency,
                           docstring='The frequency of the internal CLKsync in Hz')

        for i in [0, 1, 2, 3, 4, 5, 6, 7]:
            self.add_parameter('pxi_trigger_number {}'.format(i),
                               label='pxi trigger number {}'.format(i),
                               get_cmd=self.get_pxi_trigger(4000+i),
                               docstring='The digital value of pxi trigger no. {}, 0 (ON) of 1 (OFF)'.format(i))

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

    def off(self):
        """
        Stops the AWGs and sets the waveform of all channels to 'No Signal'
        """

        for i in [0, 1, 2, 3]:
            awg_response = self.awg.AWGstop(i)
            if (isinstance(awg_response, int) and awg_response<0):
                raise Exception('Error in call to Signadyne AWG '
                                'error code {}'.format(awg_response))
            channel_response = self.awg.channelWaveShape(i, -1)
            if (isinstance(channel_response, int) and channel_response<0):
                raise Exception('Error in call to Signadyne AWG '
                                'error code {}'.format(channel_response))

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
