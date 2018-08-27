from functools import partial
import numpy as np
import logging
from typing import Sequence, Union, Any, Tuple, Iterable
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

    def __init__(self, parent: 'N52xxBase', channel: int, name: str,
                 trace_type: str, upload_to_instrument: bool = True) -> None:

        super().__init__(parent, name)
        self._channel = channel
        self._trace_type = trace_type

        if upload_to_instrument:
            self._upload_to_instrument()

        self.add_parameter(
            'format',
            get_cmd=f'CALC{self._channel}:FORM?',
            set_cmd=f'CALC{self._channel}:FORM {{}}',
            vals=Enum(*list(self.data_formats.values()))
        )

        for format_name, format_string in self.data_formats.items():
            setattr(
                self, format_name, partial(self._get_raw_data,
                                           format_str=format_string)
            )

    def select(self) -> None:
        # Writing self.write here will cause an infinite recursion
        self.parent.write(f"CALC{self._channel}:PAR:SEL {self.short_name}")

    def write(self, cmd: str) -> None:
        self.select()
        super().write(cmd)

    def ask(self, cmd: str) -> str:
        self.select()
        return super().ask(cmd)

    def delete(self) -> None:
        self.parent.write(f'CALC{self._channel}:PAR:DEL {self.short_name}')
        channel = self.parent.channel[self._channel - 1]
        channel.delete_trace_from_list(self.short_name)

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

        self.format(format_str)
        data = np.array(visa_handle.query_binary_values(
            f'CALC{self._channel}:DATA? FDATA', datatype='f', is_big_endian=True
        ))

        return data

    def _upload_to_instrument(self) -> None:
        # PS. Do not do self.write as self.select will not work yet
        self.parent.write(
            f'CALC{self._channel}:PAR:EXT {self.short_name}, {self._trace_type}'
        )

    def __repr__(self):
        return f"{self.short_name}, {self._trace_type}"


class N52xxChannel(InstrumentChannel):
    def __init__(self, parent: 'N52xxBase', channel: int):
        super().__init__(parent, f"channel{channel}")

        self.channel = channel

        # Load the traces from the instrument
        self.trace = {
            name: N52xxTrace(
                parent, self.channel, name, trace_type,
                upload_to_instrument=False)
            for name, trace_type in self._find_traces()
        }

    def add_trace(self, name: str, tr_type: str) -> 'N52xxTrace':

        if re.search("S(.)(.)$", tr_type) is None:
            raise ValueError(
                "The trace type needs to be in the form Sxy where "
                "'x' and 'y' are integers")

        trace = N52xxTrace(self.parent, self.channel, name, tr_type)
        self.trace[name] = trace
        return trace

    def delete_trace(self, name: str) ->None:
        """
        Deletes the trace on the instrument, then calls `delete_trace_from_list`
        """
        self.trace[name].delete()

    def delete_trace_from_list(self, name: str) ->None:
        """
        Deletes the trace from the dictionary of traces. Although there is no
        leading underscore in the method name, do not call this method directly.
        Using `delete_trace` will call the delete method on the trace object
        which in turn will:
        1) Delete the trace from the instrument
        2) Call this method.
        """
        del self.trace[name]

    def delete_all_traces(self) ->None:
        trace_names = list(self.trace.keys())
        for name in trace_names:
            self.trace[name].delete()

    def _find_traces(self) -> Iterable:
        """
        Find traces on the instrument
        """
        result = self.ask(f"CALC{self.channel}:PAR:CAT:EXT?")
        if result == "NO CATALOG":
            return []

        trace_data = result.strip("\"").split(",")

        for name, trace_type in zip(trace_data[::2], trace_data[1::2]):
            yield name, trace_type


class N52xxBase(VisaInstrument):
    """
    Base qcodes driver for Agilent/Keysight series PNAs
    http://na.support.keysight.com/pna/help/latest/Programming/GP-IB_Command_Finder/SCPI_Command_Tree.htm
    """

    min_freq: float = None
    max_freq: float = None
    min_power: float = None
    max_power: float = None
    number_of_channels: int = None

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
            vals=Numbers(0, 1e6)
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

        self._channels = ChannelList(self, "channel", N52xxChannel)
        self.add_submodule("channel", self._channels)
        for count in range(self.number_of_channels):
            channel = N52xxChannel(self, channel=count+1)
            self.channel.append(channel)

        self.connect_message()

    def delete_all_traces(self) ->None:
        # Do not do this by writing "CALC:PAR:DEL:ALL" to the instrument, as
        # a reference to the trace objects will still be contained in the
        # internal dictionary of the channel object
        for channel in self.channel:
            channel.delete_all_traces()

    def run_sweep(self, averages: int =1) ->None:

        if averages == 1:
            self.sweep_mode('SING')
        else:
            self.write('SENS:AVER:CLE')
            self.write(f'SENS:SWE:GRO:COUN {averages}')
            self.sweep_mode('GRO')

        try:
            # Once the sweep mode is in hold, we know we're done
            # Note that if no triggers are received, we can get stuck in an
            # infinite loop
            while self.sweep_mode() != 'HOLD':
                time.sleep(0.1)
        except KeyboardInterrupt:
            # If the user aborts because (s)he is stuck in the infinite loop
            # mentioned above, provide a hint of what can be wrong.
            msg = "User abort detected. "
            source = self.root_instrument.trigger_source()

            if source == "MAN":
                msg += "The trigger source is manual. Are you sure this is " \
                       "correct? Please set the correct source with the " \
                       "'trigger_source' parameter"
            elif source == "EXT":
                msg += "The trigger source is external. Is the trigger " \
                       "source functional?"

            logger.warning(msg)
