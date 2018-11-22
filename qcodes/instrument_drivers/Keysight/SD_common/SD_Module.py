import warnings
from typing import Callable, List, Union
import numpy as np
from functools import partial, wraps

from qcodes.instrument.base import Instrument, Parameter
from qcodes.instrument.channel import InstrumentChannel
import qcodes.utils.validators as validators

import logging
logger = logging.getLogger(__name__)


try:
    import keysightSD1
except ImportError:
    raise ImportError('to use the Keysight SD drivers install the keysightSD1 module '
                      '(http://www.keysight.com/main/software.jspx?ckey=2784055)')


def error_check(value, method_name=None):
    """Check if returned value after a set is an error code or not.

    Args:
        value: value to test.
        method_name: Name of called SD method, used for error message

    Raises:
        AssertionError if returned value is an error code.
    """
    assert isinstance(value, (str, bool, np.ndarray)) or (int(value) >= 0), \
        f'Error in call to SD_Module.{method_name}, error code {value}'


def with_error_check(fun):
    @wraps(fun)
    def error_check_wrapper(*args, **kwargs):
        value = fun(*args, **kwargs)
        error_check(value, f'Error calling {fun.__name__} with args {kwargs}. '
                           f'Return value = {value}')
        return value
    return error_check_wrapper

def result_parser(value, name='result', verbose=False):
    """ Deprecated by error_check
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
    if isinstance(value, np.ndarray) or isinstance(value, str) or isinstance(value, bool) or (int(value) >= 0):
        if isinstance(value, np.ndarray):
            logger.debug(f'{name}: {len(value)} pts')
        else:
            logger.debug(f'{name}: {value}')

        if verbose:
            print(f'{name}: {value}')
        return value
    else:
        raise Exception(f'Error in call to SD_Module error code {value}')


class SignadyneParameter(Parameter):
    """Signadyne parameter designed to send keysightSD1 commands.

    This parameter can function as a standard parameter, but can also be
    associated with a specific keysightSD1 function.

    Args:
        name: Parameter name
        parent: Signadyne Instrument or instrument channel
            In case of a channel, it should have an id between 1 and n_channels.
        get_cmd: Standard optional Parameter get function
        get_function: keysightSD1 function to be called when getting the
            parameter value. If set, get_cmd must be None.
        set_cmd: Standard optional Parameter set function
        set_function: keysightSD1 function to be called when setting the
            parameter value. If set, set_cmd must be None.
        set_args: Optional ancillary parameter names that are passed to
            set_function. Used for some keysightSD1 functions need to pass
            multiple parameters simultaneously. If set, the name of this
            parameter must also be included in the appropriate index.
        initial_value: initial value for the parameter. This does not actually
            perform a set, so it does not call any keysightSD1 function.
        **kwargs: Additional kwargs passed to Parameter
    """
    def __init__(self,
                 name: str,
                 parent: Union[Instrument, InstrumentChannel] = None,
                 get_cmd: Callable = None,
                 get_function: Callable = None,
                 set_cmd: Callable = False,
                 set_function: Callable = None,
                 set_args: List[str] = None,
                 initial_value=None,
                 **kwargs):
        self.get_cmd = get_cmd
        self.get_function = get_function

        self.set_cmd = set_cmd
        self.set_function = set_function
        self.set_args = set_args

        super().__init__(name=name, parent=parent, **kwargs)

        if initial_value is not None:
            # We set the initial value here to ensure that it does not call
            # the set_raw method the first time
            if self.val_mapping is not None:
                initial_value = self.val_mapping[initial_value]
            self._save_val(initial_value)

    def set_raw(self, val):
        if self.set_cmd is not False:
            if self.set_cmd is not None:
                return self.set_cmd(val)
            else:
                return
        elif self.set_function is not None:
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
                        set_vals.append(getattr(self.parent, set_arg).raw_value)

            # Evaluate the set function with the necessary set parameter values
            if isinstance(self.parent, InstrumentChannel):
                # Also pass the channel id
                return_val = self.set_function(self.parent.id, *set_vals)
            else:
                return_val = self.set_function(*set_vals)
            # Check if the returned value is an error
            method_name = self.set_function.__func__.__name__
            error_check(return_val, method_name=method_name)
        else:
            # Do nothing, value is saved
            pass

    def get_raw(self):
        if self.get_cmd is not None:
            return self.get_cmd()
        elif self.get_function is not None:
            if isinstance(self.parent, InstrumentChannel):
                return self.get_function(self.parent.id)
            else:
                return self.get_function()
        else:
            return self.get_latest(raw=True)


class SD_Module(Instrument):
    """
    This is the general SD_Module driver class that implements shared parameters and functionality among all PXIe-based
    digitizer/awg/combo cards by Keysight.

    This driver was written to be inherited from by either the SD_AWG, SD_DIG or SD_Combo class, depending on the
    functionality of the card.

    Specifically, this driver was written with the M3201A and M3300A cards in mind.

    This driver makes use of the Python library provided by Keysight as part of the SD1 Software package (v.2.01.00).
    """

    def __init__(self, name, model, chassis, slot, triggers, **kwargs):
        super().__init__(name, **kwargs)

        self.model = model

        # Create instance of keysight SD_Module class
        self.SD_module = keysightSD1.SD_Module()

        self.n_triggers = triggers

        # Open the device, using the specified chassis and slot number
        module_name = self.SD_module.getProductNameBySlot(chassis, slot)
        if isinstance(module_name, str):
            result_code = self.SD_module.openWithSlot(module_name, chassis, slot)
            if result_code <= 0:
                raise Exception('Could not open SD_Module error code {result_code}')
        else:
            raise Exception('No SD Module found at chassis {chassis}, slot {slot}')

        for i in range(triggers):
            self.add_parameter('pxi_trigger_number_{}'.format(i),
                               label='pxi trigger number {}'.format(i),
                               get_cmd=partial(self.get_pxi_trigger, pxi_trigger=(4000 + i)),
                               set_cmd=partial(self.set_pxi_trigger, pxi_trigger=(4000 + i)),
                               docstring='The digital value of pxi trigger no. {}, 0 (ON) of 1 (OFF)'.format(i),
                               vals=validators.Enum(0, 1))

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
            value (int): Digital value with negated logic, 0 (ON) or 1 (OFF),
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
