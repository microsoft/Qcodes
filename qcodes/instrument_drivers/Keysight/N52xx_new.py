from functools import partial
import numpy as np
import logging
from typing import Any, Callable
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
                 trace_type: str) -> None:

        super().__init__(parent, name)
        self._channel = channel
        self._trace_type = trace_type

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
        self.select()
        data = np.array(visa_handle.query_binary_values(
            f'CALC{self._channel}:DATA? FDATA', datatype='f', is_big_endian=True
        ))

        return data

    def upload_to_instrument(self) -> None:
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

        self.add_parameter(  # per channel
            'power',
            label='Power',
            get_cmd=f'SOUR{self.channel}:POW?',
            get_parser=float,
            set_cmd=f'SOUR{self.channel}:POW {{:.2f}}',
            unit='dBm',
            vals=Numbers(
                min_value=self.parent.min_power,
                max_value=self.parent.max_power
            )
        )

        self.add_parameter(  # Per channel
            'if_bandwidth',
            label='IF Bandwidth',
            get_cmd=f'SENS{self.channel}:BAND?',
            get_parser=float,
            set_cmd=f'SENS{self.channel}:BAND {{:.2f}}',
            unit='Hz',
            vals=Numbers(min_value=1, max_value=15e6)
        )

        self.add_parameter(  # Per channel
            'averages_enabled',
            label='Averages Enabled',
            get_cmd=f"SENS{self.channel}:AVER?",
            set_cmd=f"SENS{self.channel}:AVER {{}}",
            val_mapping={True: '1', False: '0'}
        )

        self.add_parameter(  # Per channel
            'averages',
            label='Averages',
            get_cmd=f'SENS{self.channel}:AVER:COUN?',
            get_parser=int,
            set_cmd=f'SENS{self.channel}:AVER:COUN {{:d}}',
            unit='',
            vals=Numbers(min_value=1, max_value=65536)
        )

        # Setting frequency range
        self.add_parameter(
            'start',
            label='Start Frequency',
            get_cmd=f'SENS{self.channel}:FREQ:STAR?',
            get_parser=float,
            set_cmd=f'SENS{self.channel}:FREQ:STAR {{}}',
            unit='',
            vals=Numbers(
                min_value=self.parent.min_freq,
                max_value=self.parent.max_freq
            )
        )

        self.add_parameter(
            'stop',
            label='Stop Frequency',
            get_cmd=f'SENS{self.channel}:FREQ:STOP?',
            get_parser=float,
            set_cmd=f'SENS{self.channel}:FREQ:STOP {{}}',
            unit='',
            vals=Numbers(
                min_value=self.parent.min_freq,
                max_value=self.parent.max_freq
            )
        )

        self.add_parameter(
            'points',
            label='Points',
            get_cmd=f'SENS{self.channel}:SWE:POIN?',
            get_parser=int,
            set_cmd=f'SENS{self.channel}:SWE:POIN {{}}',
            unit='',
            vals=Numbers(min_value=1, max_value=100001)
        )

        self.add_parameter(
            'electrical_delay',
            label='Electrical Delay',
            get_cmd=f'CALC{self.channel}:CORR:EDEL:TIME?',
            get_parser=float,
            set_cmd=f'CALC{self.channel}:CORR:EDEL:TIME {{:.6e}}',
            unit='s',
            vals=Numbers(min_value=0, max_value=100000)
        )

        self.add_parameter(
            'sweep_time',
            label='Time',
            get_cmd=f'SENS{self.channel}:SWE:TIME?',
            get_parser=float,
            unit='s',
            vals=Numbers(0, 1e6)
        )

        self.add_parameter(
            'sweep_mode',
            label='Mode',
            get_cmd=f'SENS{self.channel}:SWE:MODE?',
            set_cmd=f'SENS{self.channel}:SWE:MODE {{}}',
            vals=Enum("HOLD", "CONT", "GRO", "SING")
        )

    @property
    def trace(self) ->dict:
        """
        Load the traces from the instrument

        Returns:
            dict: keys are trace names, values are instance of `N52xxTrace`
        """
        result = self.ask(f"CALC{self.channel}:PAR:CAT:EXT?")
        if result == "NO CATALOG":
            return {}

        trace_info = result.strip("\"").split(",")
        trace_names = trace_info[::2]
        trace_types = trace_info[1::2]

        return {
            name: N52xxTrace(
                self.parent, self.channel, name, trace_type)
            for name, trace_type in zip(trace_names, trace_types)
        }

    def add_trace(self, name: str, tr_type: str) -> 'N52xxTrace':
        """
        Add a trace the instrument. Note that if a trace with the name given
        already exists on the instrument, this trace is simply returned without
        uploading a second trace with the same name.

        Args:
            name (str): Name of the trace to add
            tr_type (str): Currently only S-parameter types are supported, which
                have the format `Sxy` where x and y are integers.

        Returns:
            trace (N52xxTrace)
        """
        if re.search("S(.)(.)$", tr_type) is None:
            raise ValueError(
                "The trace type needs to be in the form Sxy where "
                "'x' and 'y' are integers")

        traces = self.trace
        if name in traces:
            return traces[name]

        trace = N52xxTrace(self.parent, self.channel, name, tr_type)
        trace.upload_to_instrument()
        return trace

    def delete_trace(self, name: str) ->None:
        """
        Deletes the trace on the instrument

        Args:
            name (str): Name of the trace to add
        """
        self.trace[name].delete()

    def delete_all_traces(self) ->None:
        """
        Delete all traces on this channel
        """
        for trace in self.trace.values():
            trace.delete()

    def run_sweep(self, averages: int =1, blocking: bool=True) ->None:

        if averages == 1:
            self.sweep_mode('SING')
        else:
            self.write(f'SENS{self.channel}:AVER:CLE')
            self.write(f'SENS{self.channel}:SWE:GRO:COUN {averages}')
            self.sweep_mode('GRO')

        if blocking:
            self.block_while_not_hold()

    def block_while_not_hold(self) ->None:
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
            source = self.parent.trigger_source()

            if source == "MAN":
                msg += "The trigger source is manual. Are you sure this is " \
                       "correct? Please set the correct source with the " \
                       "'trigger_source' parameter"
            elif source == "EXT":
                msg += "The trigger source is external. Is the trigger " \
                       "source functional?"

            logger.warning(msg)


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

        for param in ["power", "start", "stop", "points"]:
            self.add_parameter(
                param,
                set_cmd=self._call_all_channels(name=param),
                get_cmd=self._call_all_channels(name=param)
            )

        self.connect_message()

    def _call_all_channels(self, name) ->Callable:
        """
        Perform a function call on all channels at the same time.
        Note: the ChannelList ostensibly supports this already, but this
        support is incomplete and/or broken:
            1) Class methods are not passed through to the individual channels.
               You have to create QCoDeS functions
            2) MultiParameter are returned which a) will be deprecated and b)
               does not support `set`
        """
        def caller(name: str) -> Callable:
            def inner(*args, **kwargs) ->dict:
                result = {}
                for channel_number, channel in enumerate(self.channel):
                    if len(channel.trace) == 0:
                        continue
                    channel_result = getattr(channel, name)(*args, **kwargs)
                    result[channel_number] = channel_result

                return result
            return inner

        return caller(name)

    def delete_all_traces(self) ->None:
        self.write("CALC:PAR:DEL:ALL")

    def run_sweep(self, averages: int =1):
        self._call_all_channels("run_sweep")(averages=averages, blocking=False)
        self._call_all_channels("block_while_not_hold")()
