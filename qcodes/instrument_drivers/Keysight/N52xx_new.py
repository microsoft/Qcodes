from functools import partial
import numpy as np
import logging
from typing import Sequence, Union, Any, Tuple
import time
import re

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Numbers, Enum, Bool

logger = logging.getLogger()


class N52xxTrace(InstrumentChannel):
    """
    Allow operations on individual PNA traces.
    """

    data_formats = {
        "log_magnitude": "MLOG",
        "linear_magnitude": "MLIN",
        "phase": "PHAS",
        "unwrapped_phase": "UPH",
        "group_delay": "GDEL",
        "real": "REAL",
        "imaginary": "IMAG"
    }

    def __init__(self, parent: 'N52xxBase', channel: int, trace: int) -> None:

        super().__init__(parent, name=f"channel{channel},trace{trace}")

        self._channel = channel
        self._trace = trace

        self.add_parameter(
            'format',
            get_cmd='CALC:FORM?',
            set_cmd='CALC:FORM {}',
            vals=Enum(*list(self.data_formats.values()))
        )

        for format_name, format_string in self.data_formats.items():
            setattr(
                self, format_name, partial(self._get_raw_data,
                                           format_string=format_string)
            )

    def select(self) -> None:
        self.write(f"CALC:PAR:MNUM {self._trace}")

    def _get_raw_data(self, format_str: str) -> np.ndarray:
        """
        Args:
            format_str (str): Our data is complex values (that is, has a
            magnitude and phase). Possible value are
                * "MLOG" (log_magnitude)
                * "MLIN" (linear magnitude)
                * "PHAS" (phase)
                * "UPH" (unwrapped phase)
                * "GDEL" (group delay)
                * "REAL"
                * "IMAG"
        """
        visa_handle = self.parent.visa_handle

        self.select()
        self._instrument.write(f'CALC{self._channel}:FORM {format_str}')
        data = np.array(visa_handle.query_binary_values(
            f'CALC{self._channel}:DATA? FDATA',datatype='f', is_big_endian=True
        ))

        return data


class N52xxBase(VisaInstrument):
    """
    Base qcodes driver for Agilent/Keysight series PNAs
    http://na.support.keysight.com/pna/help/latest/Programming/GP-IB_Command_Finder/SCPI_Command_Tree.htm
    """

    min_freq: float = None
    max_freq: float = None
    min_power: float = None
    max_power: float = None

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:

        super().__init__(name, address, terminator='\n', **kwargs)

        self.add_parameter(
            'power',
            label='Power',
            get_cmd='SOUR:POW?',
            get_parser=float,
            set_cmd='SOUR:POW {:.2f}',
            unit='dBm',
            vals=Numbers(min_value=self.min_power,max_value=self.max_power)
        )

        self.add_parameter(
            'if_bandwidth',
            label='IF Bandwidth',
            get_cmd='SENS:BAND?',
            get_parser=float,
            set_cmd='SENS:BAND {:.2f}',
            unit='Hz',
            vals=Numbers(min_value=1,max_value=15e6)
        )

        self.add_parameter(
            'averages_enabled',
            label='Averages Enabled',
            get_cmd="SENS:AVER?",
            set_cmd="SENS:AVER {}",
            val_mapping={True: '1', False: '0'}
        )

        self.add_parameter(
            'averages',
            label='Averages',
            get_cmd='SENS:AVER:COUN?',
            get_parser=int,
            set_cmd='SENS:AVER:COUN {:d}',
            unit='',
            vals=Numbers(min_value=1, max_value=65536)
        )

        # Setting frequency range
        self.add_parameter(
            'start',
            label='Start Frequency',
            get_cmd='SENS:FREQ:STAR?',
            get_parser=float,
            set_cmd='SENS:FREQ:STAR {}',
            unit='',
            vals=Numbers(min_value=self.min_freq, max_value=self.max_freq)
        )

        self.add_parameter(
            'stop',
            label='Stop Frequency',
            get_cmd='SENS:FREQ:STOP?',
            get_parser=float,
            set_cmd='SENS:FREQ:STOP {}',
            unit='',
            vals=Numbers(min_value=self.min_freq, max_value=self.max_freq)
        )

        # Number of points in a sweep
        self.add_parameter(
            'points',
            label='Points',
            get_cmd='SENS:SWE:POIN?',
            get_parser=int,
            set_cmd='SENS:SWE:POIN {}',
            unit='',
            vals=Numbers(min_value=1, max_value=100001)
        )

        self.add_parameter(
            'electrical_delay',
            label='Electrical Delay',
            get_cmd='CALC:CORR:EDEL:TIME?',
            get_parser=float,
            set_cmd='CALC:CORR:EDEL:TIME {:.6e}',
            unit='s',
            vals=Numbers(min_value=0, max_value=100000)
        )

        self.add_parameter(
            'sweep_time',
            label='Time',
            get_cmd='SENS:SWE:TIME?',
            get_parser=float,
            unit='s',
            vals=Numbers(0,1e6)
        )

        self.add_parameter(
            'sweep_mode',
            label='Mode',
            get_cmd='SENS:SWE:MODE?',
            set_cmd='SENS:SWE:MODE {}',
            vals=Enum("HOLD", "CONT", "GRO", "SING")
        )

        self.add_parameter(
            'active_trace',
            label='Active Trace',
            get_cmd="CALC:PAR:MNUM?",
            get_parser=int,
            set_cmd="CALC:PAR:MNUM {}",
            vals=Numbers(min_value=1, max_value=24)
        )

        self.add_parameter(
            "trigger_source",
            get_cmd="TRIG:SOUR?",
            set_cmd="TRIG:SOUR {}",
            vals=Enum("EXT", "IMM", "INT", "MAN"),
            set_parser=lambda value: "IMM" if value is "INT" else value
        )

        self._traces = ChannelList(self, "traces", N52xxTrace)
        self.add_submodule("traces", self._traces)

        self.connect_message()