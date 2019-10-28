"""
    A class is defined for each submodule, e.g. class 'Keysight_34934A' is for module '34934A'.
    A dictionary, whose keys are the module names, and values are the corresponding class, is
    defined at the end of the file.
    The dictionary should be imported in the system framework.
"""
import re
from qcodes import VisaInstrument, InstrumentChannel
from typing import Union, List, Tuple, Callable, cast


class Keysight_34933A(InstrumentChannel):
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

        super().__init__(parent, name)
        self.slot = slot

    @staticmethod
    def show_content():
        print('this is an example class')


class Keysight_34934A(InstrumentChannel):
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

        super().__init__(parent, name)

        self.add_parameter(name='protection_mode',
                           get_cmd=self._get_relay_protection_mode,
                           set_cmd=self._set_relay_protection_mode,
                           docstring='relay protection mode.')
        self.slot = slot
        configuration = self.ask(f'SYSTEM:MODule:TERMinal:TYPE? {self.slot}')
        if configuration is None:
            raise SystemError('no configuration module connected,'
                              'or safety interlock jumper removed')
        self.row, self.column = [int(num) for num in re.findall(r'\d+', configuration)]

    def _get_relay_protection_mode(self):
        return self.ask(f'SYSTem:MODule:ROW:PROTection? {self.slot}')

    def _set_relay_protection_mode(self, resistance: int = 100):
        """
        set the relay protection mode between 'AUTO100' and 'AUTO0'. 'AUTO100' has 100 Ohm
        resistance for each row. For 'AUTO0' mode, 100 Ohm is placed momentarily then bypassed,
        so no resistance afterwards.

        Args:
            resistance: either 100 or 0 for 'AUTO100' or 'AUTO0' mode, respectively

        """
        if resistance not in [0, 100]:
            raise ValueError('please input 100 or 0 for AUTO100 or AUTO0 mode')
        self.write(f'SYSTem:MODule:ROW:PROTection {self.slot}, AUTO{resistance}')

    def validate_value(self, row, column):
        return (row <= self.row) and (column <= self.column)

    def to_channel_list(self, paths: List[Tuple[int, int]]):
        """
        convert the (row, column) pair to a 4-digit channel number 'sxxx', where s is the slot
        number, xxx is generated from the numbering function.

        Args:
            paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]

        Returns:
            in the format of '(@sxxx, sxxx, sxxx, sxxx)', where sxxx is a 4-digit channel number
        """
        slot = self.slot
        numbering_function = self.numbering_function()
        channel_list = []
        for row, column in paths:
            if not self.validate_value(row, column):
                raise ValueError('input/output value out of range for current module')
            channel = f'{slot}{numbering_function(row, column)}'
            channel_list.append(channel)
        channel_list = f"(@{','.join(channel_list)})"
        return channel_list

    def is_open(self, row: int, column: int) -> bool:
        """
        to check if a channel is open/disconnected

        Args:
            row (int): row number
            column (int): column number

        Returns:
            True if the channel is open/disconnected, false if is closed/connected.
        """
        if not self.validate_value(row, column):
            raise ValueError('input/output value out of range')
        channel = self.to_channel_list([(row, column)])
        message = self.ask(f'ROUT:OPEN? {channel}')
        return bool(int(message[0]))

    def is_closed(self, row: int, column: int) -> bool:
        """
        to check if a channel is closed/connected

        Args:
            row (int): row number
            column (int): column number

        Returns:
            True if the channel is closed/connected, false if is open/disconnected.
        """
        if not self.validate_value(row, column):
            raise ValueError('input/output value out of range')
        channel = self.to_channel_list([(row, column)])
        message = self.ask(f'ROUT:CLOSe? {channel}')
        return bool(int(message[0]))

    def connect_path(self, row: int, column: int) -> None:
        """
        to connect/close the specified channels

        Args:
            row (int): row number
            column (int): column number
        """
        if not self.validate_value(row, column):
            raise ValueError('input/output value out of range for current module')
        channel = self.to_channel_list([(row, column)])
        self.write(f'ROUT:CLOSe {channel}')

    def disconnect_path(self, row: int, column: int) -> None:
        """
        to disconnect/open the specified channels

        Args:
            row (int): row number
            column (int): column number
        """
        if not self.validate_value(row, column):
            raise ValueError('input/output value out of range for current module')
        channel = self.to_channel_list([(row, column)])
        self.write(f'ROUT:OPEN {channel}')

    def connect_paths(self, paths: List[Tuple[int, int]]) -> None:
        """
        to connect/close the specified channels.

        Args:
            paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
        """
        channel_list_str = self.to_channel_list(paths)
        print(channel_list_str)
        self.write(f"ROUTe:CLOSe {channel_list_str}")

    def disconnect_paths(self, paths: List[Tuple[int, int]]) -> None:
        """
        to disconnect/open the specified channels.

        Args:
            paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
        """
        channel_list_str = self.to_channel_list(paths)
        print(channel_list_str)
        self.write(f"ROUTe:OPEN {channel_list_str}")

    def numbering_function(self):
        """
        to select the correct numbering function based on the matrix configuration
        See P168 of the user's guide for Agilent 34934A High Density Matrix Module:
        http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf

        Return:
            numbering function to get 3-digit channel number from row and column number
        """
        matrix_size = f'{self.row}x{self.column}'
        if matrix_size == '4x32':
            return self.rc2channel_number_4x32
        if matrix_size == '4x64':
            return self.rc2channel_number_4x64
        if matrix_size == '4x128':
            return self.rc2channel_number_4x128
        if matrix_size == '8x32':
            return self.rc2channel_number_8x32
        if matrix_size == '8x64':
            return self.rc2channel_number_8x64
        if matrix_size == '16x32':
            return self.rc2channel_number_16x32

    @staticmethod
    def rc2channel_number_4x32(row: int, column: int, one_wire_matrices: str) -> str:
        """
        34934A module channel numbering for 4x32 matrix setting
        see P168 of the user's guide: http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf

        Args:
            row (int): row number
            column (int): column number
            one_wire_matrices: 1 wire matrices

        Returns:
            xxx: 3-digit channel number
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

    @staticmethod
    def rc2channel_number_4x64(row: int, column: int, two_wire_matrices: str) -> str:
        """
        34934A module channel numbering for 4x64 matrix setting
        see P168 of the user's guide: http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf

        Args:
            row (int): row number
            column (int): column number
            two_wire_matrices: 1 wire matrices

        Returns:
            'xxx': 3-digit channel number
        """
        if two_wire_matrices == 'MH':
            xxx = 100*(2*row - 1) + column
        elif two_wire_matrices == 'ML':
            xxx = 100*(2*row - 1) + column + 64
        else:
            raise ValueError('Wrong value of 2 wire matrices (MH, ML)')
        return str(xxx)

    @staticmethod
    def rc2channel_number_4x128(row: int, column: int) -> str:
        """
        34934A module channel numbering for 4x128 matrix setting
        see P168 of the user's guide: http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf

        Args:
            row (int): row number
            column (int): column number

        Returns:
            'xxx': 3-digit channel number
        """
        xxx = 100*(2*row - 1) + column
        return str(xxx)

    @staticmethod
    def rc2channel_number_8x32(row: int, column: int, two_wire_matrices: str) -> str:
        """
        34934A module channel numbering for 8x32 matrix setting
        see P168 of the user's guide: http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf

        Args:
            row (int): row number
            column (int): column number
            two_wire_matrices: 1 wire matrices

        Returns:
            'xxx': 3-digit channel number
        """
        if two_wire_matrices == 'MH':
            xxx = 100*row + column
        elif two_wire_matrices == 'ML':
            xxx = 100*row + column + 32
        else:
            raise ValueError('Wrong value of 2 wire matrices (MH, ML)')
        return str(xxx)

    @staticmethod
    def rc2channel_number_8x64(row: int, column: int) -> str:
        """
        34934A module channel numbering for 8x64 matrix setting
        see P168 of the user's guide: http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf

        Args:
            row (int): row number
            column (int): column number

        Returns:
            'xxx': 3-digit channel number
        """
        xxx = 100*row + column
        return str(xxx)

    @staticmethod
    def rc2channel_number_16x32(row: int, column: int) -> str:
        """
        34934A module channel numbering for 16x32 matrix setting
        see P168 of the user's guide: http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf

        Args:
            row (int): row number
            column (int): column number

        Returns:
            'xxx': 3-digit channel number
        """
        xxx = 50*(row + 1) + column
        return str(xxx)


Keysight_model = {'34933A': Keysight_34933A, '34934A': Keysight_34934A}
