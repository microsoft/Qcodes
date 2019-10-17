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

    :param func: function to wrap
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


def module_dictionary(module_name: str) -> classmethod:  # is "classmethod" correct?
    if module_name == '34934A':
        return Keysight_34934A
    if module_name == '34933A':
        return Keysight_34933A
    # need to add more once available


def identical_list(elements: list):
    if all(element == elements[0] for element in elements):
        return True
    return False


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
        # super().__init__(name, address, terminator='\n', **kwargs)

        self.add_parameter(name='system_slots_info',
                           get_cmd=self._system_slots_info,
                           docstring='getting the information of slots being used.')

        self.add_parameter(name='get_status',
                           get_cmd='*ESR?',
                           get_parser=int,
                           docstring='Queries status register.')

        self.add_parameter(name='get_error',
                           get_cmd=':SYST:ERR?',
                           docstring='Queries error queue')

        self.occupied_slots = sorted(self.system_slots_info())
        self.rows = np.array([])          # need to remove in the future
        self.columns = np.array([])       # need to remove in the future
        self.modules_in_slot = dict.fromkeys(self.occupied_slots)
        self.connect_message()
        self.scan_slots()

    def scan_slots(self):
        system_slots_info = self.system_slots_info()
        for slot in self.occupied_slots:
            module, matrix_size = system_slots_info[slot].split('-')
            print(f'slot {slot}: module-{module}, matrix-{matrix_size}')
            row_size, column_size = matrix_size.split('x')
            self.rows = np.append(self.rows, int(row_size))
            self.columns = np.append(self.columns, int(column_size))
            if module_dictionary(module) is None:
                raise ValueError(f'unknown module {module}')
            self.modules_in_slot[slot] = module_dictionary(module)(row_size, column_size)

    def _slot_bins(self) -> Tuple:      # need to remove in the future, config file will handle this
        rows = self.rows
        columns = self.columns
        if (not identical_list(rows)) and (not identical_list(columns)):
            print('each module is a matrix by itself, can not be connected together')
            # this is actually not totally true, it's possible that some, but not all, of the modules are connects
            connected = 'individually'
            return rows, columns, connected
        elif identical_list(rows) and (not identical_list(columns)):
            connected = 'by_rows'  # it's also possible the modules are not connected with each other at all
        elif (not identical_list(rows)) and identical_list(columns):
            connected = 'by_columns'  # same as above
        else:
            connected = 'by_rows'  # also not true, because the system won't be able to tell
        return np.cumsum(rows), np.cumsum(columns), connected

    def convert_row_and_column(self, row, column):
        rows, columns, connected = self._slot_bins()
        slots_indices = self.occupied_slots
        idx = 0
        if connected == 'individually':
            return slots_indices[0], row, column
        if (connected == 'by_rows') and (column > columns[0]):
            idx = next(i for i, value in enumerate(columns) if value > column)
            column = column - columns[idx - 1]
        if (connected == 'by_columns') and (row > rows[0]):
            idx = next(i for i, value in enumerate(rows) if value > row)
            row = row - rows[idx - 1]
        return slots_indices[idx], int(row), int(column)

    def to_channel_list(self, paths: List[Tuple[int, int]]):
        """
        equations are different for different model and matrices setting
        need to be defined and imported from python modules for each model
        :param paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
        :return: in the format of (@sxxx, sxxx, sxxx, sxxx), where sxxx is a 4-digit channel number
        """
        channel_list = []
        for row, column in paths:
            slot, row_new, column_new = self.convert_row_and_column(row, column)
            numbering_function = self.modules_in_slot[slot].numbering_function()
            channel = f'{slot}{numbering_function(row_new, column_new)}'
            channel_list.append(channel)
        channel_list = f"(@{','.join(channel_list)})"
        return channel_list

    @post_execution_status_poll
    def _system_slots_info(self) -> dict:
        """
        the command CYST:CTYP? returns the following:
        Agilent	Technologies,<Model	Number>,<Serial	Number>, <Firmware	Rev>
        where <Model Number> is '0' if there is no model connected
        :return: a dictionary, with slot number as the key, and model number the value
        """
        slots_dict = {}
        for i in range(1, 9):
            modules = self.ask(f'SYST:CTYP? {i}').split(',')
            if modules[1] != '0':
                slots_dict[i] = modules[1]
        return slots_dict

    @post_execution_status_poll
    def connect_path(self, input_ch: int, output_ch: int) -> None:
        """
        to connect two channels
        >>> connect_path(input_ch=2, output_ch=5)
        :param input_ch: input channel, usually the row number
        :param output_ch: output channel, usually the column number
        :return: None
        """
        connection = [(input_ch, output_ch)]
        channel_str = self.to_channel_list(connection)
        self.write(f"ROUTe:CLOSe {channel_str}")

    @post_execution_status_poll
    def disconnect_path(self, input_ch: int, output_ch: int) -> None:
        """
        to disconnect two channels
        >>> disconnect_path(input_ch=2, output_ch=5)
        :param input_ch: input channel, usually the row number
        :param output_ch: output channel, usually the column number
        :return: None
        """
        connection = [(input_ch, output_ch)]
        channel_str = self.to_channel_list(connection)
        self.write(f"ROUTe:OPEN {channel_str}")

    @post_execution_status_poll
    def is_closed(self, input_ch: int, output_ch: int) -> bool:
        """
        to check if two channels are connected
        >>> is_closed(input_ch=2, output_ch=5)
        :param input_ch: input channel, usually the row number
        :param output_ch: output channel, usually the column number
        :return: True if two channels are connected/closed, false if they are open.
        """
        connection = [(input_ch, output_ch)]
        channel_str = self.to_channel_list(connection)
        message = self.ask(f'ROUT:CLOS? {channel_str}')
        return bool(int(message[0]))

    @post_execution_status_poll
    def is_open(self, input_ch: int, output_ch: int) -> bool:
        """
        to check if two channels are open
        >>> is_open(input_ch=2, output_ch=5)
        :param input_ch: input channel, usually the row number
        :param output_ch: output channel, usually the column number
        :return: True if two channels are open, false if they are connected/closed.
        """
        connection = [(input_ch, output_ch)]
        channel_str = self.to_channel_list(connection)
        message = self.ask(f'ROUT:OPEN? {channel_str}')
        return bool(int(message[0]))

    @post_execution_status_poll
    def connect_paths(self, paths: List[Tuple[int, int]]) -> None:
        """
        to connect/close the specified channels	on a switch	module.
        :param paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
        :return: None
        """
        channel_list_str = self.to_channel_list(paths)
        print(channel_list_str)
        self.write(f"ROUTe:CLOSe {channel_list_str}")

    @post_execution_status_poll
    def disconnect_paths(self, paths: List[Tuple[int, int]]) -> None:
        """
        to disconnect/open the specified channels on a switch module.
        :param paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
        :return: None
        """
        channel_list_str = self.to_channel_list(paths)
        self.write(f"ROUT:OPEN {channel_list_str}")

    @post_execution_status_poll
    def disconnect_all(self, slot='') -> None:
        """
        to disconnect/open all connections on select
        :param slot: slot number, between 1 and 8, default value is empty, which means all slots
        :return: None
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
