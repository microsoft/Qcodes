"""
Base qcodes driver for Agilent/Keysight series PNAs
http://na.support.keysight.com/pna/help/latest/Programming/GP-IB_Command_Finder/SCPI_Command_Tree.htm
"""
from functools import partial
import numpy as np
import logging
from typing import Any
import time
import re

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Numbers, Enum, MultiType, Union

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
        """
        Upload the trace to the instrument
        """
        # Do not do self.write; self.select will not work yet as the instrument
        # has not been uploaded yet
        self.parent.write(
            f'CALC{self._channel}:PAR:EXT {self.short_name}, {self._trace_type}'
        )

    def __repr__(self):
        return f"{self.short_name}, {self._trace_type}"


class N52xxChannel(InstrumentChannel):
    """
    Allows operations on specific channels.
    """
    def __init__(self, parent: 'N52xxBase', channel: int, description: str):
        super().__init__(parent, f"channel{channel}")

        self.channel = channel
        self.description = description

        self.add_parameter(
            'power',
            label='Power',
            get_cmd=f'SOUR{self.channel}:POW?',
            get_parser=float,
            set_cmd=f'SOUR{self.channel}:POW {{:.2f}}',
            unit='dBm',
            vals=Numbers(
                min_value=self.parent.min_power,
                max_value=self.parent.max_power
            ),
            set_parser=float
        )

        self.add_parameter(
            'if_bandwidth',
            label='IF Bandwidth',
            get_cmd=f'SENS{self.channel}:BAND?',
            get_parser=float,
            set_cmd=f'SENS{self.channel}:BAND {{:.2f}}',
            unit='Hz',
            vals=Numbers(min_value=1, max_value=15e6)
        )

        self.add_parameter(
            'averages_enabled',
            label='Averages Enabled',
            get_cmd=f"SENS{self.channel}:AVER?",
            set_cmd=f"SENS{self.channel}:AVER {{}}",
            val_mapping={True: '1', False: '0'}
        )

        self.add_parameter(
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
            label='Frequency Points',
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
            # Make sure we are in stepped sweep
            set_cmd=f'SENS{self.channel}:SWE:TIME {{:.6e}}',
            unit='s',
            vals=Numbers(0, 1e6)
        )

        self.add_parameter(
            'dwell_time',
            label='Time',
            get_cmd=f'SENS{self.channel}:SWE:DWEL?',
            get_parser=float,
            # Make sure we are in stepped sweep
            set_cmd=f'SENS{self.channel}:SWE:GEN STEP;'
                    f'SENS{self.channel}:SWE:DWEL {{:.6e}}',
            unit='s',
            vals=MultiType(Numbers(0, 1e6), Enum("min", "max"))
        )

        self.add_parameter(
            'sweep_mode',
            label='Mode',
            get_cmd=f'SENS{self.channel}:SWE:MODE?',
            set_cmd=f'SENS{self.channel}:SWE:MODE {{}}',
            vals=Enum("HOLD", "CONT", "GRO", "SING")
        )

        self.add_parameter(
            "sensor_correction",
            get_cmd=f"SENS{self.channel}:CORR?",
            set_cmd=f"SEND{self.channel}:CORR {{}}",
            vals=Enum(0, 1, "0", "1", True, False),
            set_parser=int
        )

    @property
    def trace(self) ->dict:
        """
        Interface to access traces on the instrument

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
        trace.select()
        return trace

    def delete_trace(self, name: str) ->None:
        """
        Deletes the trace on the instrument

        Args:
            name (str): Either the name of the trace to delete or "all"
        """
        if name == "all":
            for trace in self.trace.values():
                trace.delete()
        else:
            self.trace[name].delete()

    def run_sweep(self, averages: int =1, blocking: bool=True) ->None:
        """
        Run a sweep

        Args:
            averages (int): The number of averages
            blocking (bool): If True, this method will block until the sweep
                                has finished
        """
        if averages == 1:
            self.sweep_mode('SING')
        else:
            self.write(f'SENS{self.channel}:AVER:CLE')
            self.write(f'SENS{self.channel}:SWE:GRO:COUN {averages}')
            self.sweep_mode('GRO')

        if blocking:
            self.block_while_not_hold()

    def block_while_not_hold(self) ->None:
        """
        Block until a sweep has finished
        """
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

    def get_snp_data(self, ports: list=None) ->np.ndarray:
        """
        Extract S-parameter data in snp format. The 'n' in 'snp' stands for an
        integer. For instance, s4p stands for scatter parameters
        of a four port device (Scattering 4 Port = s4p).

        For each frequency in the measurement sweep the snp data consists of a
        complex n-by-n matrix. This command returns SNP data without header
        information, and in columns, not in rows as .SnP files. This means that
        the data returned from this command sends all frequency data, then all
        Sx1 magnitude or real data, then all Sx1 phase or imaginary data, and
        so forth.

        For each frequency, the length of the data array is
        therefore 2 * n**2 + 1

        For more information about the snp format, please visit:
        http://literature.cdn.keysight.com/litweb/pdf/ads2004a/cktsim/ck04a8.html

        Note: we could do the same by reading data from individual traces.
        However this method uses a single instrument query to retrieve all the
        data

        Args:
            ports (list): The ports from which we want data (e.g. [1, 2, 3, 4])

        Returns:
            data (ndarray): Array of length

                (2 * n**2 + 1) * n_freq

            where:
             * n is the number of ports requested
             * n_freq is the number of frequency points in the sweep

        Example:
            Please view:
            qcodes/docs/examples/driver_examples/Qcodes_example_with_Keysight_PNA_N5222B.ipynb
        """
        if ports is None:
            ports_string = "1,2,3,4"
        else:
            ports_string = ",".join([str(p) for p in ports])

        # We want our SNP data in Real-Imaginary format
        self.write('MMEM:STOR:TRAC:FORM:SNP RI')
        write_string = f'CALC{self.channel}:DATA:SNP:PORT? "{ports_string}"'
        data = np.array(
            self.parent.visa_handle.query_binary_values(
                write_string,  datatype='f', is_big_endian=True
            )
        )
        self.parent.synchronize()
        return data

    def __repr__(self):
        return f"Channel class: {self.description}"


class N52xxPort(InstrumentChannel):
    """
    Allow operations on individual PNA ports.
    """

    def __init__(
            self,
            parent: 'N52xxBase',
            name: str,
            port: int,
            min_power: Union[int, float],
            max_power: Union[int, float]
    ) -> None:

        super().__init__(parent, name)

        self.port = int(port)
        if self.port not in range(1, 5):
            raise ValueError("Port must be between 1 and 4.")

        self.add_parameter(
            "source_power",
            label="power",
            unit="dBm",
            get_cmd=f"SOUR:POW{self.port}?",
            set_cmd=f"SOUR:POW{self.port} {{}}",
            get_parser=float,
            vals=Numbers(min_value=min_power,max_value=max_power)
        )


class N52xxBase(VisaInstrument):
    """
    TODO: Proper docstring
    """

    min_freq: float = None
    max_freq: float = None
    min_power: float = None
    max_power: float = None
    port_count: int = None

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:

        super().__init__(name, address, terminator='\n', **kwargs)
        self.active_channel: N52xxChannel = None

        self.add_parameter(
            "trigger_source",
            get_cmd="TRIG:SOUR?",
            set_cmd="TRIG:SOUR {}",
            vals=Enum("EXT", "IMM", "INT", "MAN"),
            set_parser=lambda value: "IMM" if value is "INT" else value
        )

        self.add_parameter(
            "display_arrangement",
            set_cmd="DISP:ARR {}",
            vals=Enum("TILE", "CASC", "OVER", "STAC", "SPL", "QUAD")
        )

        self._channels = ChannelList(self, "channel", N52xxChannel)
        self.add_submodule("channel", self._channels)
        self.add_channel("default")

        # Ports
        ports = ChannelList(self, "port", N52xxPort)
        for port_num in range(1, self.port_count + 1):
            port = N52xxPort(
                self, f"port{port_num}", port_num, self.min_power,
                self.max_power
            )

            ports.append(port)
            self.add_submodule(f"port{port_num}", port)

        ports.lock()
        self.add_submodule("port", ports)

        self.connect_message()

    def add_channel(self, description):
        if description in [c.description for c in self._channels]:
            raise ValueError(
                f"Channel description {description} already exists")

        channel_count = len(self._channels)

        channel = N52xxChannel(
            self, channel=channel_count + 1, description=description)

        self.channel.append(channel)
        self.active_channel = channel

    def activate_channel(self, description):
        channel = {c.description: c for c in self._channels}[description]
        self.active_channel = channel

    def list_channels(self):
        print("\n".join(self._channels))

    def delete_all_traces(self) ->None:
        """
        Delete all traces from the instrument. Note that this is different then

        >>> pna = N52xxBase("pna", 'GPIB0::16::INSTR')
        >>> pna.delete_trace("all")

        As this will only delete from the first channel
        """
        self.write("CALC:PAR:DEL:ALL")

    def synchronize(self):
        self.ask("*OPC?")

    def reset_instrument(self):
        self.write("*RST")
        self.write("*CLS")
        # sane settings
        self.write('FORM REAL,32')
        self.write('FORM:BORD NORM')
        self.trigger_source("IMM")

    def __getattr__(self, item):
        """
        Map the attributes of the channel class on the base instrument
        """
        try:
            return super().__getattr__(item)
        except AttributeError:
            if item == "active_channel":
                # We have produced an unwanted recursion
                raise

            att = getattr(self.active_channel, item, None)

            if att is None:
                raise AttributeError(
                    f"No attribute {item} on instrument or channel class")

            return att
