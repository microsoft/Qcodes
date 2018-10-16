"""
Base instrument module for the Keysight N52xx instrument series
For documentation, see:
http://na.support.keysight.com/pna/help/latest/Programming/GP-IB_Command_Finder/SCPI_Command_Tree.htm
"""

import logging
from typing import Union, Any, Callable

from ._N52xx_channel_ext import N52xxChannelList, N52xxInstrumentChannel
from .channel import N52xxChannel
from .trace import N52xxTrace
from qcodes import Instrument, VisaInstrument, ChannelList, InstrumentChannel
from qcodes.utils.validators import Enum, Numbers

logger = logging.getLogger()


class N52xxPort(InstrumentChannel):
    """
    Allow operations on individual N52xx ports.
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
            vals=Numbers(min_value=min_power, max_value=max_power)
        )


class N52xxWindow(N52xxInstrumentChannel):
    """
    Windows on the instrument refer to user interface elements on the instrument
    display which can independently show different measurements
    """
    discover_command = "SYST:WIND:CAT?"
    max_trace_count = 24

    def __init__(
            self, parent: Instrument, identifier: Any, existence: bool = False,
            channel_list: 'N52xxChannelList' = None, **kwargs) -> None:

        super().__init__(
            parent, identifier=f"window{identifier}", existence=existence,
            channel_list=channel_list, **kwargs
        )

        self._window = identifier
        self._trace_count = 0

    def _create(self) ->None:
        self.base_instrument.write(f"DISP:WINDow{self._window} ON")

    def _delete(self) ->None:
        self.parent.write(f"DISP:WINDow{self._window} OFF")

    def add_trace(self, trace: N52xxTrace) ->None:
        """
        Add a trace to the window
        """
        trace_number = self._trace_count + 1

        if trace_number > self.max_trace_count:
            raise RuntimeError("Maximum number of traces in this window "
                               "exceeded")

        trace_name = trace.short_name
        self.parent.write(
            f"DISP:WIND{self._window}:TRAC{trace_number}:FEED '{trace_name}'")

        self._trace_count += 1


class N52xxBase(VisaInstrument):
    """
    Base class instrument module for the Keysight N52xx instrument series
    """

    min_freq: float = None
    max_freq: float = None
    min_power: float = None
    max_power: float = None
    port_count: int = None

    def __init__(self, name: str, address: str, **kwargs) ->None:

        super().__init__(name, address, terminator='\n', **kwargs)

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

        self.base_instrument = self

        # Add ports
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

        # Windows, measurements and channels are QCoDeS channel types which can
        # be added and deleted on the instrument. This is unlike ports, which
        # have a fixed number depending on the instrument type
        channels = N52xxChannelList(
            parent=self, name="channels", chan_type=N52xxChannel
        )
        self.add_submodule("channels", channels)

        windows = N52xxChannelList(
            parent=self, name="windows", chan_type=N52xxWindow
        )
        self.add_submodule("windows", windows)

        self.connect_message()

    def synchronize(self) ->None:
        """
        This is sometimes needed when a parallel SCPI command is called, to
        ensure we wait until command completion
        """
        self.ask("*OPC?")

    def reset_instrument(self) ->None:
        """
        Put the instrument back into a sane state
        """
        self.write("*RST")
        self.write("*CLS")
        self.write('FORM REAL,32')
        self.write('FORM:BORD NORM')
        self.trigger_source("IMM")

    def _raw_io(self, cmd: str, io_function: Callable) ->str:
        """
        Print error messages from the instrument (if any) if the python
        logger is in debug mode
        """
        ret = io_function(cmd)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            err = super().ask_raw('SYST:ERR?')
            if not err.startswith("+0"):
                logger.warning(f"Command {cmd} produced error {err}")

        return ret

    def write_raw(self, cmd):
        return self._raw_io(cmd, super().write_raw)

    def ask_raw(self, cmd):
        return self._raw_io(cmd, super().ask_raw)
