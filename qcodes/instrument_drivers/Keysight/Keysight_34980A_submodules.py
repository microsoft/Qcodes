from qcodes import VisaInstrument, InstrumentChannel
from typing import Union, Callable, cast


class Keysight_34933A(InstrumentChannel):
    """
    Create an instance of the module.
    Args:
        name (str): Name used by QCoDeS. Appears in the DataSet
    """
    def __init__(
            self,
            parent: Union[VisaInstrument, InstrumentChannel],
            name: str,
            row: int,
            column: int
    ) -> None:

        super().__init__(parent, name)
        self.row = row
        self.column = column

    @staticmethod
    def show_content():
        print('this is an empty class')


class Keysight_34934A(InstrumentChannel):
    """
    Create an instance of the module.
    Args:
        name (str): Name used by QCoDeS. Appears in the DataSet
    """
    def __init__(
            self,
            parent: Union[VisaInstrument, InstrumentChannel],
            name: str,
            row: int,
            column: int
    ) -> None:

        super().__init__(parent, name)
        self.row = row
        self.column = column
        self.slot = self.name[-1]
        self.set_relay_protection_mode(0)

    def get_relay_protection_mode(self):
        print(self.ask(f'SYSTem:MODule:ROW:PROTection? {self.slot}'))
        return self.ask(f'SYSTem:MODule:ROW:PROTection? {self.slot}')

    def set_relay_protection_mode(self, resistance: int = 100):
        """

        :param resistance:
        :return:
        """
        print('current protection mode is')
        self.get_relay_protection_mode()
        print(f'now change it to {resistance} Ohm mode')
        self.write(f'SYSTem:MODule:ROW:PROTection {self.slot}, AUTO{resistance}')
        self.get_relay_protection_mode()

    def numbering_function(self):
        """
        to obtain the numbering function for each 34934A module installed, based on
        the matrix configuration of each module
        :return: a function
        """
        return self.channel_numbering_table()

    def channel_numbering_table(self):
        """
        to select the correct numbering function based on the matrix configuration
        See P168 of the user's guide for Agilent 34934A High Density Matrix Module:
        http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf
        :param matrix_size: matrix size of the module installed
        :return: numbering function to get 3-digit channel number from row and column number
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
