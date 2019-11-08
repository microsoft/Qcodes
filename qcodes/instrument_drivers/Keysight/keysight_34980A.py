import logging
import warnings
from functools import wraps
from .keysight_34980A_submodules import KEYSIGHT_MODELS
from qcodes import VisaInstrument
from typing import List, Callable, Optional

LOGGER = logging.getLogger()


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
        super().__init__(name, address, terminator='\n', **kwargs)

        self.add_parameter(name='get_status',
                           get_cmd='*ESR?',
                           get_parser=int,
                           docstring='Queries status register.')

        self.add_parameter(name='get_error',
                           get_cmd=':SYST:ERR?',
                           docstring='Queries error queue')

        self._system_slots_info_dict: Optional[dict] = None
        self.module = dict.fromkeys(self.system_slots_info.keys())
        self.scan_slots()
        self.connect_message()

    def scan_slots(self) -> None:
        """
        Scan the occupied slots and make an object for each switch matrix
        module installed
        """
        for slot in self.system_slots_info.keys():
            module_info = self.system_slots_info[slot]['module']
            for module in KEYSIGHT_MODELS:
                if module in module_info:
                    name = 'slot' + str(slot)
                    sub_mod = KEYSIGHT_MODELS[module](self, name, slot)
                    self.module[slot] = sub_mod
                    break
            if self.module[slot] is None:
                self.module[slot] = 'Unknown: ' + module_info
                logging.warning(f'unknown module in {module_info}')

    @property
    def system_slots_info(self):
        if self._system_slots_info_dict is None:
            self._system_slots_info_dict = self._system_slots_info()
        return self._system_slots_info_dict

    @post_execution_status_poll
    def _system_slots_info(self) -> dict:
        """
        the command SYST:CTYP? returns the following:
        Agilent	Technologies,<Model Number>,<Serial Number>,<Firmware Rev>
        where <Model Number> is '0' if there is no model connected

        Returns:
            a dictionary, with slot numbers as the keys, and model numbers
            the values
        """
        slots_dict = {}
        keys = ['vendor', 'module', 'serial', 'firmware']
        for i in range(1, 9):
            identity = self.ask(f'SYST:CTYP? {i}').strip('"').split(',')
            if identity[1] != '0':
                slots_dict[i] = dict(zip(keys, identity))
        return slots_dict

    @post_execution_status_poll
    def _are_closed(self, channel: str) -> List[bool]:
        """
        to check if a channel is closed/connected

        Args:
            channel (str): example: '(@1203)' for channel between row=2,
                                    column=3 in slot 1
                                    '(@sxxx, sxxx, sxxx)' for multiple channels

        Returns:
            a list of True and/or False
            True if the channel is closed/connected
            False if is open/disconnected.
        """
        messages = self.ask(f'ROUT:CLOSe? {channel}')
        return [bool(int(message)) for message in messages.split(',')]

    @post_execution_status_poll
    def _are_open(self, channel: str) -> List[bool]:
        """
        to check if a channel is open/disconnected

        Args:
            channel (str): example: '(@1203)' for channel between row=2,
                                    column=3 in slot 1
                                    '(@sxxx, sxxx, sxxx)' for multiple channels

        Returns:
            a list of True and/or False
            True if the channel is open/disconnected
            False if is closed/connected.
        """
        messages = self.ask(f'ROUT:OPEN? {channel}')
        return [bool(int(message)) for message in messages.split(',')]

    @post_execution_status_poll
    def _connect_paths(self, channel_list: str) -> None:
        """
        to connect/close the specified channels	on a switch	module.

        Args:
            channel_list: in the format of '(@sxxx, sxxx, sxxx, sxxx)',
                        where sxxx is a 4-digit channel number
        """
        self.write(f"ROUTe:CLOSe {channel_list}")

    @post_execution_status_poll
    def _disconnect_paths(self, channel_list: str) -> None:
        """
        to disconnect/open the specified channels on a switch module.

        Args:
            channel_list: in the format of '(@sxxx, sxxx, sxxx, sxxx)',
                        where sxxx is a 4-digit channel number
        """
        self.write(f"ROUT:OPEN {channel_list}")

    @post_execution_status_poll
    def disconnect_all(self, slot='') -> None:
        """
        to open/disconnect all connections on select module

        Args:
            slot (int): slot number, between 1 and 8, default value is
                    ''(empty), which means all slots
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
