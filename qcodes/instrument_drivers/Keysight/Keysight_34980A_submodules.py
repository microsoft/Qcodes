"""
    A class is defined for each submodule, e.g. class 'Keysight_34934A' is for module '34934A'.
    A dictionary, whose keys are the module names, and values are the corresponding class, is
    defined at the end of the file.
    The dictionary should be imported in the system framework.
"""
import logging
import warnings
from functools import wraps
import re
import numpy as np
from qcodes import VisaInstrument, InstrumentChannel, validators
from typing import Union, List, Tuple, Optional, Callable

logger = logging.getLogger()


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


class KeysightSubModule(InstrumentChannel):
    """
    Create an instance for submodule for the 34980A system.
    Args:
        parent: the system which the module is installed on
        name: user defined name for the module
        slot: the slot the module is installed
    """
    def __init__(
            self,
            parent: Union[VisaInstrument, InstrumentChannel],
            name: str,
            slot: int
    ) -> None:

        super().__init__(parent, name)

        self.add_parameter(name='get_status',
                           get_cmd='*ESR?',
                           get_parser=int,
                           docstring='Queries status register.')

        self.add_parameter(name='get_error',
                           get_cmd=':SYST:ERR?',
                           docstring='Queries error queue')

        self.slot = slot

    def validate_value(self, row: int, column: int) -> None:
        """
        to check if the row and column number is within the range of the module layout.

        Args:
            row (int): row value
            column (int): column value
        """
        raise NotImplementedError("Please subclass this")

    def to_channel_list(self, paths: List[Tuple[int, int]], wiring_config: Optional[str] = None) -> str:
        """
        convert the (row, column) pair to a 4-digit channel number 'sxxx', where s is the slot
        number, xxx is generated from the numbering function.
        This may be different for different modules.

        Args:
            paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
            wiring_config (str): for 1-wire matrices, values are 'MH', 'ML';
                              for 2-wire matrices, values are 'M1H', 'M2H', 'M1L', 'M2L'

        Returns:
            in the format of '(@sxxx, sxxx, sxxx, sxxx)', where sxxx is a 4-digit channel number
        """
        raise NotImplementedError("Please subclass this")

    @post_execution_status_poll
    def is_open(self, row: int, column: int) -> bool:
        """
        to check if a channel is open/disconnected

        Args:
            row (int): row number
            column (int): column number

        Returns:
            True if the channel is open/disconnected, false if it's closed/connected.
        """
        self.validate_value(row, column)
        channel = self.to_channel_list([(row, column)])
        message = self.ask(f'ROUT:OPEN? {channel}')
        return bool(int(message))

    @post_execution_status_poll
    def is_closed(self, row: int, column: int) -> bool:
        """
        to check if a channel is closed/connected

        Args:
            row (int): row number
            column (int): column number

        Returns:
            True if the channel is closed/connected, false if it's open/disconnected.
        """
        self.validate_value(row, column)
        channel = self.to_channel_list([(row, column)])
        message = self.ask(f'ROUT:CLOSe? {channel}')
        return bool(int(message))

    @post_execution_status_poll
    def connect_path(self, row: int, column: int) -> None:
        """
        to connect/close the specified channels

        Args:
            row (int): row number
            column (int): column number
        """
        self.validate_value(row, column)
        channel = self.to_channel_list([(row, column)])
        self.write(f'ROUT:CLOSe {channel}')

    @post_execution_status_poll
    def disconnect_path(self, row: int, column: int) -> None:
        """
        to disconnect/open the specified channels

        Args:
            row (int): row number
            column (int): column number
        """
        self.validate_value(row, column)
        channel = self.to_channel_list([(row, column)])
        self.write(f'ROUT:OPEN {channel}')

    @post_execution_status_poll
    def connect_paths(self, paths: List[Tuple[int, int]]) -> None:
        """
        to connect/close the specified channels.

        Args:
            paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
        """
        channel_list_str = self.to_channel_list(paths)
        self.write(f"ROUTe:CLOSe {channel_list_str}")

    @post_execution_status_poll
    def disconnect_paths(self, paths: List[Tuple[int, int]]) -> None:
        """
        to disconnect/open the specified channels.

        Args:
            paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
        """
        channel_list_str = self.to_channel_list(paths)
        self.write(f"ROUTe:OPEN {channel_list_str}")

    @post_execution_status_poll
    def are_closed(self, paths: List[Tuple[int, int]]) -> List[bool]:
        """
        to check if a list of channels is closed/connected

        Args:
            paths: list of channels [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]

        Returns:
            True if the channel is closed/connected, false if it's open/disconnected.
        """
        channel_list_str = self.to_channel_list(paths)
        messages = self.ask(f"ROUTe:CLOSe? {channel_list_str}")
        return [bool(int(message)) for message in messages.split(',')]

    @post_execution_status_poll
    def are_open(self, paths: List[Tuple[int, int]]) -> List[bool]:
        """
        to check if a list of channels is open/disconnected

        Args:
            paths: list of channels [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]

        Returns:
            True if the channel is closed/connected, false if it's open/disconnected.
        """
        channel_list_str = self.to_channel_list(paths)
        messages = self.ask(f"ROUTe:OPEN? {channel_list_str}")
        return [bool(int(message)) for message in messages.split(',')]

    def clear_status(self) -> None:
        """
        Clears status register and error queue of the instrument.
        """
        self.write('*CLS')


class Keysight_34934A(KeysightSubModule):
    """
    Create an instance for module 34933A.
    Args:
        parent: the system which the module is installed on
        name: user defined name for the module
        slot: the slot the module is installed
    """
    def __init__(
            self,
            parent: Union[VisaInstrument, InstrumentChannel],
            name: str,
            slot: int
    ) -> None:

        super().__init__(parent, name, slot)

        self.add_parameter(name='protection_mode',
                           get_cmd=self._get_relay_protection_mode,
                           set_cmd=self._set_relay_protection_mode,
                           valus=validators.Enum('AUTO100', 'AUTO0', 'FIX', 'ISO'),
                           docstring='get and set relay protection mode.')
        self.slot = slot
        configuration = self.ask(f'SYSTEM:MODule:TERMinal:TYPE? {self.slot}')
        self._is_locked = (configuration == 'NONE')
        if self._is_locked:
            logging.warning(f'For slot {slot}, no configuration module connected, '
                            f'or safety interlock jumper removed.')
        else:
            self.row, self.column = [int(num) for num in re.findall(r'\d+', configuration)]

    def write(self, cmd: str):
        if self._is_locked:
            logging.warning("Warning: no configuration module connected, "
                            "or safety interlock enabled")
            return

        return super().write(cmd)

    def validate_value(self, row: int, column: int) -> None:
        """
        to check if the row and column number is within the range of the module layout.

        Args:
            row (int): row value
            column (int): column value
        """
        if (row > self.row) or (column > self.column):
            raise ValueError('row/column value out of range')
        return

    @post_execution_status_poll
    def _get_relay_protection_mode(self):
        return self.ask(f'SYSTem:MODule:ROW:PROTection? {self.slot}')

    @post_execution_status_poll
    def _set_relay_protection_mode(self, mode: str = 'AUTO100'):
        """
        set the relay protection mode. The fastest switching speeds for relays in a given
        signal path are achieved using the FIXed or ISOlated modes, followed by the AUTO100
        and AUTO0 modes. There may be a maximum of 200 Ohm of resistance, which can only be
        bypassed by "AUTO0" mode.
        See manual and programmer's reference for detailed explanation.

        Args:
            mode: names for protections modes
        """
        self.write(f'SYSTem:MODule:ROW:PROTection {self.slot}, {mode}')

    def to_channel_list(self, paths: List[Tuple[int, int]], wiring_config: Optional[str] = None) -> str:
        """
        convert the (row, column) pair to a 4-digit channel number 'sxxx', where s is the slot
        number, xxx is generated from the numbering function.

        Args:
            paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
            wiring_config (str): for 1-wire matrices, values are 'MH', 'ML';
                              for 2-wire matrices, values are 'M1H', 'M2H', 'M1L', 'M2L'

        Returns:
            in the format of '(@sxxx, sxxx, sxxx, sxxx)', where sxxx is a 4-digit channel number
        """
        layout = f'{self.row}x{self.column}'
        numbering_function = self.get_numbering_function(layout, wiring_config)
        channel_list = []
        for row, column in paths:
            self.validate_value(row, column)
            channel = f'{self.slot}{numbering_function(row, column)}'
            channel_list.append(channel)
        channel_list = f"(@{','.join(channel_list)})"
        return channel_list

    @staticmethod
    def get_numbering_function(layout, wiring_config=None):
        """
        to select the correct numbering function based on the matrix configuration.
        On P168 of the user's guide for Agilent 34934A High Density Matrix Module:
        http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf
        there are eleven equations. This function here simplifies them to one.

        Args:
            layout (str): the layout of the matrix module, e.g. 4x32
            wiring_config (str): wiring configuration for 1 or 2 wired matrices

        Returns:
            The numbering function to convert row and column in to a 3-digit number
        """
        available_layouts = {
            "4x32": ["M1H", "M2H", "M1L", "M2L"],
            "4x64": ["MH", "ML"],
            "4x128": [None],
            "8x32": ["MH", "ML"],
            "8x64": [None],
            "16x32": [None]
        }

        if layout not in available_layouts:
            raise ValueError(f"Unsupported layout: {layout}")

        if wiring_config not in available_layouts[layout]:
            raise ValueError(
                f"Invalid wiring config '{wiring_config}' for layout {layout}"
            )

        offsets = {
            "M1H": 0,
            "M2H": 1,
            "M1L": 2,
            "M2L": 3,
            "MH": 0,
            "ML": 1
        }

        rows, columns = np.array(layout.split("x")).astype(int)

        offset = 0
        if wiring_config is not None:
            offset = offsets[wiring_config] * columns

        a = 800 / rows
        offset += 100 - a

        def numbering_function(row, col):
            return str(int(a * row + col + offset))

        return numbering_function


keysight_models = {'34934A': Keysight_34934A}
