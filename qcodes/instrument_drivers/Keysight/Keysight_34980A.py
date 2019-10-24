# QCoDeS driver for the Keysight 34980A Multifunction Switch/Measure Unit
import re
import numpy as np
import warnings
from functools import wraps
from .Keysight_34980A_submodules import Keysight_34934A, Keysight_34933A
from qcodes import VisaInstrument
from qcodes.utils.validators import MultiType, Ints, Enum, Lists
from typing import List, Tuple, Callable
from pyvisa import VisaIOError


def post_execution_status_poll(func: Callable) -> Callable:
    """
    Generates a decorator that clears the instrument's status registers
    before executing the actual call and reads the status register after the
    function call to determine whether an error occurs.

    Args:
        func: function to wrap
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.clear_status()
        retval = func(self, *args, **kwargs)

        stb = self.get_status()
        if stb:
            warnings.warn(f"Instrument status byte indicates an error occurred "
                          f"(value of STB was: {stb})! Use `get_error` method "
                          f"to poll error message.",
                          stacklevel=2)
        return retval

    return wrapper


def module_dictionary(module_name: str) -> classmethod:
    """
    Return the driver for selected module used on the 34980A system. The drivers are loaded from
    the Keysight_34980A_submodules. New driver needs to be added to the submodules before use.

    Args:
        module_name (str): name of the module used on the 34980A system
    """
    if module_name == '34934A':
        return Keysight_34934A
    if module_name == '34933A':
        return Keysight_34933A


class Keysight_34980A(VisaInstrument):
    """
    QCodes driver for 34980A switch/measure unit
    """
    def __init__(self, name, address, **kwargs):
        """
        Create an instance of the instrument.

        Args:
            name (str): Name used by QCoDeS. Appears in the DataSet
            address (str): Visa-resolvable instrument address.
        """
        super().__init__(name, address, **kwargs)

        self.add_parameter(name='get_status',
                           get_cmd='*ESR?',
                           get_parser=int,
                           docstring='Queries status register.')

        self.add_parameter(name='get_error',
                           get_cmd=':SYST:ERR?',
                           docstring='Queries error queue')

        self.occupied_slots = sorted(self.system_slots_info)
        self.module_in_slot = dict.fromkeys(self.occupied_slots)
        self.scan_slots()
        self.connect_message()

    def scan_slots(self) -> None:
        """
        Scan the occupied slots and make an object for each switch matrix module installed
        """
        system_slots_info = self.system_slots_info
        for slot in self.occupied_slots:
            module, matrix_size = system_slots_info[slot].split('-')
            print(f'slot {slot}: module-{module}, matrix-{matrix_size}')
            row_size, column_size = matrix_size.split('x')

            if module_dictionary(module) is None:
                raise ValueError(f'unknown module {module}')
            name = 'slot' + str(slot)
            self.add_submodule(
                name,
                module_dictionary(module)(
                    self, name, int(row_size), int(column_size), slot
                )
            )
            self.module_in_slot[slot] = eval(f'self.{name}')

    @property
    def system_slots_info(self):
        return self._system_slots_info()

    @post_execution_status_poll
    def _system_slots_info(self) -> dict:
        """
        the command SYST:CTYP? returns the following:
        Agilent	Technologies,<Model Number>,<Serial Number>,<Firmware Rev>
        where <Model Number> is '0' if there is no model connected

        Returns:
            a dictionary, with slot number as the key, and model number the value
        """
        slots_dict = {}
        for i in range(1, 9):
            modules = self.ask(f'SYST:CTYP? {i}').split(',')
            if modules[1] != '0':
                slots_dict[i] = modules[1]
        return slots_dict

    @post_execution_status_poll
    def is_closed(self, channel: str) -> bool:
        """
        to check if a channel is closed/connected

        Args:
            channel (str): example: '(@1203)' for channel between row=2, column=3 in slot 1

        Returns:
            True if the channel is closed/connected, false if is open/disconnected.
        """
        message = self.ask(f'ROUT:CLOS? {channel}')
        return bool(int(message[0]))

    @post_execution_status_poll
    def is_open(self, channel: str) -> bool:
        """
        to check if a channel is open/disconnected

        Args:
            channel (str): example: '(@1203)' for channel between row=2, column=3 in slot 1

        Returns:
            True if the channel is open/disconnected, false if is closed/connected.
        """
        message = self.ask(f'ROUT:OPEN? {channel}')
        return bool(int(message[0]))

    @post_execution_status_poll
    def connect_paths(self, channel_list) -> None:
        """
        to connect/close the specified channels	on a switch	module.

        Args:
            channel_list: in the format of '(@sxxx, sxxx, sxxx, sxxx)', where sxxx is a
                        4-digit channel number
        """
        print(channel_list)
        self.write(f"ROUTe:CLOSe {channel_list}")

    @post_execution_status_poll
    def disconnect_paths(self, channel_list) -> None:
        """
        to disconnect/open the specified channels on a switch module.

        Args:
            channel_list: in the format of '(@sxxx, sxxx, sxxx, sxxx)', where sxxx is a
                        4-digit channel number
        """
        self.write(f"ROUT:OPEN {channel_list}")

    @post_execution_status_poll
    def disconnect_all(self, slot='') -> None:
        """
        to open/disconnect all connections on select module

        Args:
            slot (int): slot number, between 1 and 8, default value is ''(empty), which
                    means all slots
        """
        self.write(f'ROUT:OPEN:ALL {slot}')

    def clear_status(self) -> None:
        """
        Clears status register and error queue of the instrument.
        """
        self.write('*CLS')

    def reset(self) -> None:
        """
        Performs an instrument reset.
        Does not reset error queue!
        """
        self.write('*RST')
