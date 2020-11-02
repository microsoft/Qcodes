import logging
import re
from qcodes import VisaInstrument, InstrumentChannel, validators
from typing import Union, List, Tuple, Optional, Callable
from .keysight_34980a_submodules import KeysightSwitchMatrixSubModule


class Keysight34934A(KeysightSwitchMatrixSubModule):
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
                           vals=validators.Enum('AUTO100',
                                                 'AUTO0',
                                                 'FIX',
                                                 'ISO'),
                           docstring='get and set the relay protection mode.'
                                     'The fastest switching speeds for relays'
                                     'in a given signal path are achieved using'
                                     'the FIXed or ISOlated modes, followed'
                                     'by the AUTO100 and AUTO0 modes.'
                                     'There may be a maximum of 200 Ohm of'
                                     'resistance, which can only be bypassed'
                                     'by "AUTO0" mode. See manual and'
                                     'programmer''s reference for detail.')

        layout = self.ask(f'SYSTEM:MODule:TERMinal:TYPE? {self.slot}')
        self._is_locked = (layout == 'NONE')
        if self._is_locked:
            logging.warning(f'For slot {slot}, no configuration module'
                            f'connected, or safety interlock jumper removed. '
                            "Making any connection is not allowed")
            config = self.ask(f'SYST:CTYP? {slot}').strip('"').split(',')[1]
            layout = config.split('-')[1]
        self.row, self.column = [
            int(num) for num in re.findall(r'\d+', layout)
        ]

    def write(self, cmd: str) -> None:
        """
        When the module is safety interlocked, users can not make any
        connections. There will be no effect when try to connect any channels.
        """
        if self._is_locked:
            logging.warning("Warning: no configuration module connected, "
                            "or safety interlock enabled. "
                            "Making any connection is not allowed")
        return self.parent.write(cmd)

    def validate_value(self, row: int, column: int) -> None:
        """
        to check if the row and column number is within the range of the
        module layout.

        Args:
            row: row value
            column: column value
        """
        if (row > self.row) or (column > self.column):
            raise ValueError('row/column value out of range')

    def _get_relay_protection_mode(self) -> str:
        return self.ask(f'SYSTem:MODule:ROW:PROTection? {self.slot}')

    def _set_relay_protection_mode(self, mode: str) -> None:
        self.write(f'SYSTem:MODule:ROW:PROTection {self.slot}, {mode}')

    def to_channel_list(
            self,
            paths: List[Tuple[int, int]],
            wiring_config: Optional[str] = ''
    ) -> str:
        """
        convert the (row, column) pair to a 4-digit channel number 'sxxx', where
        s is the slot number, xxx is generated from the numbering function.

        Args:
            paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3)]
            wiring_config: for 1-wire matrices, values are 'MH', 'ML';
                                 for 2-wire matrices, values are 'M1H', 'M2H',
                                 'M1L', 'M2L'

        Returns:
            in the format of '(@sxxx, sxxx, sxxx, sxxx)', where sxxx is a
            4-digit channel number
        """
        numbering_function = self.get_numbering_function(
            self.row,
            self.column,
            wiring_config
        )

        channels = []
        for row, column in paths:
            channel = f'{self.slot}{numbering_function(row, column)}'
            channels.append(channel)
        channel_list = f"(@{','.join(channels)})"
        return channel_list

    @staticmethod
    def get_numbering_function(
            rows: int,
            columns: int,
            wiring_config: Optional[str] = ''
    ) -> Callable:
        """
        to select the correct numbering function based on the matrix layout.
        On P168 of the user's guide for Agilent 34934A High Density Matrix
        Module:
        http://literature.cdn.keysight.com/litweb/pdf/34980-90034.pdf
        there are eleven equations. This function here simplifies them to one.

        Args:
            rows: the total row number of the matrix module
            columns: the total column number of the matrix module
            wiring_config: wiring configuration for 1 or 2 wired matrices

        Returns:
            The numbering function to convert row and column in to a 3-digit
            number
        """
        layout = f'{rows}x{columns}'
        available_layouts = {
            "4x32": ["M1H", "M2H", "M1L", "M2L"],
            "4x64": ["MH", "ML"],
            "4x128": [''],
            "8x32": ["MH", "ML"],
            "8x64": [''],
            "16x32": ['']
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

        offset = 0
        if wiring_config != '':
            offset = offsets[wiring_config] * columns

        channels_per_row = 800 / rows
        offset += 100 - int(channels_per_row)

        def numbering_function(row: int, col: int) -> str:
            return str(int(channels_per_row * row + col + offset))

        return numbering_function
