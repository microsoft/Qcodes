import logging
import struct
import numpy as np
import warnings
from typing import List, Dict, Optional

import qcodes as qc
from qcodes import VisaInstrument, DataSet
from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument.base import Instrument, Parameter
from qcodes.instrument.parameter import ArrayParameter, ParameterWithSetpoints
import qcodes.utils.validators as vals
from qcodes.utils.helpers import create_on_off_val_mapping


class KeithleyMatrixChannel(InstrumentChannel):
    """
    """

    def __init__(self, parent: Instrument, name: str, channel: str) -> None:
        """
        """
        super().__init__(parent, name)


class Keithley_3706A(VisaInstrument):
    """
    """
    def __init__(self, name: str, address: str, **kwargs) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS
            address: VISA resource address
        """
        super().__init__(name, address, terminator='\n', **kwargs)

        self.connect_message()

    def get_idn(self) -> Dict[str, Optional[str]]:
        idnstr = self.ask_raw('*IDN?')
        vendor, model, serial, firmware = map(str.strip, idnstr.split(','))
        model = model[6:]

        idn: Dict[str, Optional[str]] = {'vendor': vendor, 'model': model,
                                         'serial': serial, 'firmware': firmware}
        return idn

    def ask(self, cmd: str) -> str:
        """
        Override of normal ask. This is important, since queries to the
        instrument must be wrapped in 'print()'
        """
        return super().ask('print({:s})'.format(cmd))
