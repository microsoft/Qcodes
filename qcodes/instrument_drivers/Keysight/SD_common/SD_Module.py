import warnings

from qcodes.instrument.base import Instrument
from numpy import ndarray

try:
    import keysightSD1
except ImportError:
    raise ImportError('to use the Keysight SD drivers install the keysightSD1 module '
                      '(http://www.keysight.com/main/software.jspx?ckey=2784055)')


def result_parser(value, name='result', verbose=False):
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
        parsed value, which is the same as value if non-negative
        or not a number
    """
    if isinstance(value, ndarray) or isinstance(value, str) or isinstance(value, bool) or (int(value) >= 0):
        if verbose:
            print('{}: {}'.format(name, value))
        return value
    else:
        raise Exception('Error in call to SD_Module '
                        'error code {}'.format(value))


class SD_Module(Instrument):
    """
    This is the general SD_Module driver class that implements shared parameters and functionality among all PXIe-based
    digitizer/awg/combo cards by Keysight.

    This driver was written to be inherited from by either the SD_AWG, SD_DIG or SD_Combo class, depending on the
    functionality of the card.

    Specifically, this driver was written with the M3201A and M3300A cards in mind.

    This driver makes use of the Python library provided by Keysight as part of the SD1 Software package (v.2.01.00).
    """

    def __init__(self, name, chassis, slot, **kwargs):
        super().__init__(name, **kwargs)

        # Create instance of keysight SD_Module class
        self.SD_module = keysightSD1.SD_Module()

        # Open the device, using the specified chassis and slot number
        module_name = self.SD_module.getProductNameBySlot(chassis, slot)
        if isinstance(module_name, str):
            result_code = self.SD_module.openWithSlot(module_name, chassis, slot)
            if result_code <= 0:
                raise Exception('Could not open SD_Module '
                                'error code {}'.format(result_code))
        else:
            raise Exception('No SD Module found at '
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

    #
    # Get-commands
    #

    def get_module_count(self, verbose=False):
        """Returns the number of SD modules installed in the system"""
        value = self.SD_module.moduleCount()
        value_name = 'module_count'
        return result_parser(value, value_name, verbose)

    def get_product_name(self, verbose=False):
        """Returns the product name of the device"""
        value = self.SD_module.getProductName()
        value_name = 'product_name'
        return result_parser(value, value_name, verbose)

    def get_serial_number(self, verbose=False):
        """Returns the serial number of the device"""
        value = self.SD_module.getSerialNumber()
        value_name = 'serial_number'
        return result_parser(value, value_name, verbose)

    def get_chassis(self, verbose=False):
        """Returns the chassis number where the device is located"""
        value = self.SD_module.getChassis()
        value_name = 'chassis_number'
        return result_parser(value, value_name, verbose)

    def get_slot(self, verbose=False):
        """Returns the slot number where the device is located"""
        value = self.SD_module.getSlot()
        value_name = 'slot_number'
        return result_parser(value, value_name, verbose)

    def get_status(self, verbose=False):
        """Returns the status of the device"""
        value = self.SD_module.getStatus()
        value_name = 'status'
        return result_parser(value, value_name, verbose)

    def get_firmware_version(self, verbose=False):
        """Returns the firmware version of the device"""
        value = self.SD_module.getFirmwareVersion()
        value_name = 'firmware_version'
        return result_parser(value, value_name, verbose)

    def get_hardware_version(self, verbose=False):
        """Returns the hardware version of the device"""
        value = self.SD_module.getHardwareVersion()
        value_name = 'hardware_version'
        return result_parser(value, value_name, verbose)

    def get_type(self, verbose=False):
        """Returns the type of the device"""
        value = self.SD_module.getType()
        value_name = 'type'
        return result_parser(value, value_name, verbose)

    def get_open(self, verbose=False):
        """Returns whether the device is open (True) or not (False)"""
        value = self.SD_module.isOpen()
        value_name = 'open'
        return result_parser(value, value_name, verbose)

    def get_pxi_trigger(self, pxi_trigger, verbose=False):
        """
        Returns the digital value of the specified PXI trigger

        Args:
            pxi_trigger (int): PXI trigger number (4000 + Trigger No.)
            verbose (bool): boolean indicating verbose mode

        Returns:
            int: Digital value with negated logic, 0 (ON) or 1 (OFF),
            or negative numbers for errors
        """
        value = self.SD_module.PXItriggerRead(pxi_trigger)
        value_name = 'pxi_trigger number {}'.format(pxi_trigger)
        return result_parser(value, value_name, verbose)

    #
    # Set-commands
    #

    def set_pxi_trigger(self, value, pxi_trigger, verbose=False):
        """
        Sets the digital value of the specified PXI trigger

        Args:
            pxi_trigger (int): PXI trigger number (4000 + Trigger No.)
            value (int): Digital value with negated logic, 0 (ON) or 1 (OFF)
        """
        result = self.SD_module.PXItriggerWrite(pxi_trigger, value)
        value_name = 'set pxi_trigger {} to {}'.format(pxi_trigger, value)
        return result_parser(result, value_name, verbose)

    #
    # FPGA related functions
    #

    def get_fpga_pc_port(self, port, data_size, address, address_mode, access_mode, verbose=False):
        """
        Reads data at the PCport FPGA Block

        Args:
            port (int): PCport number
            data_size (int): number of 32-bit words to read (maximum is 128 words)
            address (int): address that wil appear at the PCport interface
            address_mode (int): ?? not in the docs
            access_mode (int): ?? not in the docs
        """
        data = self.SD_module.FPGAreadPCport(port, data_size, address, address_mode, access_mode)
        value_name = 'data at PCport {}'.format(port)
        return result_parser(data, value_name, verbose)

    def set_fpga_pc_port(self, port, data, address, address_mode, access_mode, verbose=False):
        """
        Writes data at the PCport FPGA Block

        Args:
            port (int): PCport number
            data (array): array of integers containing the data
            address (int): address that wil appear at the PCport interface
            address_mode (int): ?? not in the docs
            access_mode (int): ?? not in the docs
        """
        result = self.SD_module.FPGAwritePCport(port, data, address, address_mode, access_mode)
        value_name = 'set fpga PCport {} to data:{}, address:{}, address_mode:{}, access_mode:{}'\
            .format(port, data, address, address_mode, access_mode)
        return result_parser(result, value_name, verbose)

    #
    # The methods below are not used for setting or getting parameters, but can be used in the test functions of the
    # test suite e.g. The main reason they are defined is to make this driver more complete
    #

    def get_product_name_by_slot(self, chassis, slot, verbose=False):
        value = self.SD_module.getProductNameBySlot(chassis, slot)
        value_name = 'product_name'
        return result_parser(value, value_name, verbose)

    def get_product_name_by_index(self, index, verbose=False):
        value = self.SD_module.getProductNameByIndex(index)
        value_name = 'product_name'
        return result_parser(value, value_name, verbose)

    def get_serial_number_by_slot(self, chassis, slot, verbose=False):
        warnings.warn('Returns faulty serial number due to error in Keysight lib v.2.01.00', UserWarning)
        value = self.SD_module.getSerialNumberBySlot(chassis, slot)
        value_name = 'serial_number'
        return result_parser(value, value_name, verbose)

    def get_serial_number_by_index(self, index, verbose=False):
        warnings.warn('Returns faulty serial number due to error in Keysight lib v.2.01.00', UserWarning)
        value = self.SD_module.getSerialNumberByIndex(index)
        value_name = 'serial_number'
        return result_parser(value, value_name, verbose)

    def get_type_by_slot(self, chassis, slot, verbose=False):
        value = self.awg.getTypeBySlot(chassis, slot)
        value_name = 'type'
        return result_parser(value, value_name, verbose)

    def get_type_by_index(self, index, verbose=False):
        value = self.awg.getTypeByIndex(index)
        value_name = 'type'
        return result_parser(value, value_name, verbose)

    #
    # The methods below are useful for controlling the device, but are not used for setting or getting parameters
    #

    # closes the hardware device and also throws away the current instrument object
    # if you want to open the instrument again, you have to initialize a new instrument object
    def close(self):
        self.SD_module.close()
        super().close()

    # only closes the hardware device, does not delete the current instrument object
    def close_soft(self):
        self.SD_module.close()

    def open_with_serial_number(self, name, serial_number):
        self.SD_module.openWithSerialNumber(name, serial_number)

    def open_with_slot(self, name, chassis, slot):
        self.SD_module.openWithSlot(name, chassis, slot)

    def run_self_test(self):
        value = self.SD_module.runSelfTest()
        print('Did self test and got result: {}'.format(value))
