from .Keysight_34980A import _Keysight_34980A, post_execution_status_poll
from pyvisa import VisaIOError
from qcodes import VisaInstrument
from qcodes.utils.validators import MultiType, Ints, Enum, Lists
from typing import List, Tuple, Callable


class Keysight_34934A(_Keysight_34980A):
    """
    This is the qcodes driver for the Keysight 34934A High Density Matrix Module
    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)

        self.add_parameter(name='module_slots_info',
                           get_cmd=self._slots_info,
                           docstring='getting the slots info for current module.')

    # @property
    def _slots_info(self) -> dict:
        """
        to get the slots where a 34934A module is connected, and the matrix size
        :return: a dictionary, with slot number as the key, and matrix size the value
        """
        slots_info = {}
        system_slots_info = self.system_slots_info()
        for key in system_slots_info:
            info = system_slots_info[key].split('-')
            if info[0] == '34934A':
                slots_info[key] = info[1]
        return slots_info

    @property
    def _numbering_function(self) -> dict:
        """
        to obtain the numbering function for each 34934A module installed, based on
        the matrix configuration of each module
        :return: a dictionary, with slot number as the key, and numbering function the value
        """
        slots_info = self.module_slots_info()
        numbering_function = {
            slot: channel_numbering_table(slots_info[slot])
            for slot in slots_info
        }
        return numbering_function

    def convert_row_and_column(self, row, column, slots_list) -> Tuple:
        """
        hmm... not a good name, because all this function does is to get the correct slots number
        :param row:
        :param column:
        :param slots_list:
        :return:
        """
        i = 0
        row0, column0 = [int(s) for s in self.module_slots_info()[slots_list[i]].split('x')]
        if (row > row0) and (column > column0):
            raise ValueError('TODO: will need a configuration file for this situation')
        while row > row0:
            row = row - row0
            i = i + 1
            if i == len(slots_list):
                raise ValueError('the row number excess the max possible value')
            row0, column0 = [int(s) for s in self.module_slots_info()[slots_list[i]].split('x')]
        while column > column0:
            column = column - column0
            i = i + 1
            if i == len(slots_list):
                raise ValueError('the column number excess the max possible value')
            row0, column0 = [int(s) for s in self.module_slots_info()[slots_list[i]].split('x')]
        slot = slots_list[i]
        return slot, row, column

    def to_channel_list(self, paths: List[Tuple[int, int]]):
        channel_list = []
        slots_list = [key for key in self.module_slots_info()]
        slots_list.sort()
        for row, column in paths:
            slot, row_new, column_new = self.convert_row_and_column(row, column, slots_list)
            channel = f'{slot}{self._numbering_function[slot](row_new, column_new)}'
            channel_list.append(channel)
        channel_list = f"(@{','.join(channel_list)})"
        return channel_list


def channel_numbering_table(matrix_size: str):
    """
    to select the correct numbering function based on the matrix configuration
    See P168 of the user's guide for Agilent 34934A High Density Matrix Module:
    http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf
    :param matrix_size: matrix size of the module installed
    :return: numbering function to get 3-digit channel number from row and column number
    """
    # matrix_size = f'{row}x{column}'
    if matrix_size == '4x32':
        return rc2channel_number_4x32
    if matrix_size == '4x64':
        return rc2channel_number_4x64
    if matrix_size == '4x128':
        return rc2channel_number_4x128
    if matrix_size == '8x32':
        return rc2channel_number_8x32
    if matrix_size == '8x64':
        return rc2channel_number_8x64
    if matrix_size == '16x32':
        return rc2channel_number_16x32


def rc2channel_number_4x32(row: int, column: int, one_wire_matrices: str) -> str:
    """
    34934A module channel numbering for 4x32 matrix setting
    see P168 of the user's guide: http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf
    :param row: row number
    :param column: column number
    :param one_wire_matrices: 1 wire matrices
    :return: 3-digit channel number
    """
    if one_wire_matrices == 'M1H':
        xxx = 100*(2*row - 1) + column
    elif one_wire_matrices == 'M2H':
        xxx = 100*(2*row - 1) + column + 32
    elif one_wire_matrices == 'M1L':
        xxx = 100*(2*row - 1) + column + 64
    elif one_wire_matrices == 'M2L':
        xxx = 100*(2*row - 1) + column + 96
    else:
        raise ValueError('Wrong value of 1 wire matrices (M1H, M1L, M2H, M2L)')
    return str(xxx)


def rc2channel_number_4x64(row: int, column: int, two_wire_matrices: str) -> str:
    """
    34934A module channel numbering for 4x64 matrix setting
    see P168 of the user's guide: http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf
    :param row: row number
    :param column: column number
    :param two_wire_matrices: 2 wire matrices
    :return: 3-digit channel number
    """
    if two_wire_matrices == 'MH':
        xxx = 100*(2*row - 1) + column
    elif two_wire_matrices == 'ML':
        xxx = 100*(2*row - 1) + column + 64
    else:
        raise ValueError('Wrong value of 2 wire matrices (MH, ML)')
    return str(xxx)


def rc2channel_number_4x128(row: int, column: int) -> str:
    """
    34934A module channel numbering for 4x128 matrix setting
    see P168 of the user's guide: http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf
    :param row: row number
    :param column: column number
    :return: 3-digit channel number
    """
    xxx = 100*(2*row - 1) + column
    return str(xxx)


def rc2channel_number_8x32(row: int, column: int, two_wire_matrices: str) -> str:
    """
        34934A module channel numbering for 8x32 matrix setting
        see P168 of the user's guide: http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf
        :param row: row number
        :param column: column number
        :param two_wire_matrices: 2 wire matrices
        :return: 3-digit channel number
        """
    if two_wire_matrices == 'MH':
        xxx = 100*row + column
    elif two_wire_matrices == 'ML':
        xxx = 100*row + column + 32
    else:
        raise ValueError('Wrong value of 2 wire matrices (MH, ML)')
    return str(xxx)


def rc2channel_number_8x64(row: int, column: int) -> str:
    """
    34934A module channel numbering for 8x64 matrix setting
    see P168 of the user's guide: http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf
    :param row: row number
    :param column: column number
    :return: 3-digit channel number
    """
    xxx = 100*row + column
    return str(xxx)


def rc2channel_number_16x32(row: int, column: int) -> str:
    """
    34934A module channel numbering for 16x32 matrix setting
    see P168 of the user's guide: http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf
    :param row: row number
    :param column: column number
    :return: 3-digit channel number
    """
    xxx = 50*(row + 1) + column
    return str(xxx)


