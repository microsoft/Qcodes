# QCoDeS driver for the Keysight 34980A Multifunction Switch/Measure Unit
import re
import warnings
from functools import wraps

from qcodes import VisaInstrument
from qcodes.utils.validators import MultiType, Ints, Enum, Lists
from typing import List, Tuple, Callable


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

        self.add_parameter(name='get_error',
                           get_cmd=':SYST:ERR?',
                           docstring='Queries error queue')

    # def to_channel_list_old(self, paths: List[Tuple[int, int]]):
    #     l = [f'{self._card:01d}{i:02d}{o:02d}' for i, o in paths]
    #     channel_list = f"(@{','.join(l)})"
    #     return channel_list

    def to_channel_list(self, paths: List[Tuple[int, int]]):
        channel_list = [f'{(c - 1) // 64 + 1:01d}{(100 * r + c//65+c%65):03d}' for r, c in paths]
        channel_list = f"(@{','.join(channel_list)})"
        return channel_list

    def connect_paths(self, paths: List[Tuple[int, int]]):
        channel_list_str = self.to_channel_list(paths)
        # self.write(f":CLOS {channel_list_str}")
        self.write(f"ROUTe:CLOSe {channel_list_str}")

    def disconnect_paths(self, paths: List[Tuple[int, int]]):
        channel_list_str = self.to_channel_list(paths)
        # self.write(f":OPEN {channel_list_str}")
        self.write(f"ROUT:OPEN {channel_list_str}")

    def disconnect_all(self):
        """
        opens all connections.

        If ground or bias mode is enabled it will connect all outputs to the
        GND or Bias Port
        """
        # self.write(f':OPEN:CARD {self._card}')
        # self.write('ROUT:OPEN:ALL [{1, 2 | ALL}]')
        self.write('ROUT:OPEN:ALL')

    def reset(self):
        """Performs an instrument reset.
        Does not reset error queue!
        """
        self.write('*RST')
