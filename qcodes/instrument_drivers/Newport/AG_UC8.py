"""
QCoDeS driver for Newport AGILIS AG-UC8 Piezo Stepper Controller.
"""

import logging
import time
from typing import Optional, Dict, Callable

import serial

from qcodes import Instrument, VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Bool, Numbers, Ints, Anything

log = logging.getLogger(__name__)


def make_get_parser(cmd: str) -> Callable[[str], int]:
    """Return a parser function which expects a line
    starting with <cmd> followed by an integer, and returns
    only the integer.
    """

    def get_parser(resp: str) -> int:
        if resp.startswith(cmd):
            try:
                return int(resp[len(cmd):].strip())
            except ValueError:
                pass
        log.warn("Unexpected response from %s: %r" % (cmd, resp))
        raise Newport_AG_UC8_Exception("Unexpected response from %s: %r" % (cmd, resp))

    return get_parser


class Newport_AG_UC8_Exception(Exception):
    pass


class Newport_AG_UC8_ErrorCode(Newport_AG_UC8_Exception):
    def __init__(self, cmd: str, err: int) -> None:
        self.failed_command = cmd
        self.error_code = err
        super().__init__("Command %s failed with error code %d" % (cmd, err))


class Newport_AG_UC8_Axis(InstrumentChannel):
    """Represents one of the 2 axes of one channel of an AG-UC8 controller."""

    def __init__(self, parent: 'Newport_AG_UC8_Channel', axis: int) -> None:
        assert axis in (1, 2)
        super().__init__(parent, "Axis %d" % axis)

        self.axis = axis

        self.add_parameter("step_delay",
                           label="Step delay in units of 10 us",
                           get_cmd="%dDL?" % axis,
                           set_cmd="%dDL{}" % axis,
                           get_parser=make_get_parser("%dDL" % axis),
                           vals=Ints(0, 200000))
        self.add_parameter("step_amplitude_pos",
                           label="Step amplitude in positive direction",
                           get_cmd="%dSU+?" % axis,
                           set_cmd="%dSU+{}" % axis,
                           get_parser=make_get_parser("%dSU+" % axis),
                           vals=Ints(1, 50))
        self.add_parameter("step_amplitude_neg",
                           label="Step amplitude in negative direction",
                           get_cmd="%dSU-?" % axis,
                           set_cmd="%dSU-{}" % axis,
                           get_parser=make_get_parser("%dSU-" % axis),
                           vals=Ints(1, 50))
        self.add_parameter("steps",
                           label="Accumulated number of steps since last reset of the step counter",
                           get_cmd="%dTP" % axis,
                           get_parser=make_get_parser("%dTP" % axis))
        self.add_parameter("status",
                           label="Status of the axis (0: ready, 1: stepping, 2: jogging, 3: moving to limit)",
                           get_cmd="%dTS" % axis,
                           get_parser=make_get_parser("%dTS" % axis))

        self.add_function("jog",
                          call_cmd="%dJA{}" % axis,
                          args=(Ints(-4, 4),))

        self.add_function("move_limit",
                          call_cmd="%dMV{}" % axis,
                          args=(Ints(-4, 4),))

        self.add_function("measure_position",
                          call_cmd=self.make_measure_position_cmd(axis),
                          args=(),
                          return_parser=make_get_parser("%dMA" % axis))

        self.add_function("move_abs",
                          call_cmd=self.make_move_abs_cmd(axis),
                          args=(Ints(0, 1000),),
                          return_parser=make_get_parser("%dPA" % axis))

        self.add_function("move_rel",
                          call_cmd="%dPR{}" % axis,
                          args=(Ints(-2147483648, 2147483647),))

        self.add_function("stop",
                          call_cmd="%dST" % axis,
                          args=())

        self.add_function("zero_position",
                          call_cmd="%dZP" % axis,
                          args=())

    def make_measure_position_cmd(self, axis: int) -> Callable[[], str]:
        """Return a call_cmd function to execute the MA command."""

        def call_cmd() -> str:
            tmo    = self.root_instrument.timeout()
            tmptmo = self.root_instrument.position_scan_timeout
            self.root_instrument.timeout(tmptmo)
            try:
                return self.ask("%dMA" % axis)
            finally:
                self.root_instrument.timeout(tmo)

        return call_cmd

    def make_move_abs_cmd(self, axis: int) -> Callable[[int], str]:
        """Return a call_cmd function to execute the PA command."""

        def call_cmd(x: int) -> str:
            tmo    = self.root_instrument.timeout()
            tmptmo = self.root_instrument.position_scan_timeout
            self.root_instrument.timeout(tmptmo)
            try:
                return self.ask("%dPA%d" % (axis, x))
            finally:
                self.root_instrument.timeout(tmo)

        return call_cmd


class Newport_AG_UC8_Channel(InstrumentChannel):
    """Represents one of the 4 channels of an AG-UC8 controller."""

    def __init__(self, parent: 'Newport_AG_UC8', channel: int) -> None:
        assert channel in (1, 2, 3, 4)
        super().__init__(parent, "Channel %d" % channel)

        self.channel = channel

        self.add_submodule("axis1", Newport_AG_UC8_Axis(self, 1))
        self.add_submodule("axis2", Newport_AG_UC8_Axis(self, 2))

        self.add_parameter("limit_status",
                           label="Limit switch status (0: not active, 1: active on axis 1, 2: active on axis 2, 3: active on both axes)",
                           get_cmd="PH",
                           get_parser=make_get_parser('PH'))

    def write(self, cmd: str) -> None:
        return self.parent.write_channel(self.channel, cmd)

    def ask(self, cmd: str) -> str:
        return self.parent.ask_channel(self.channel, cmd)


class Newport_AG_UC8(VisaInstrument):
    """
    QCoDeS driver for the Newport AGILIS AG-UC8 Piezo Stepper Controller.

    Args:
        name (str): name of the instrument.

        address (str): VISA string describing the serial port,
            for example "ASRL3" for COM3.
    """

    default_timeout = 1.0
    position_scan_timeout = 120.0
    command_delay = 0.002   # wait 2 ms between commands
    reset_delay = 0.05      # wait 50 ms after reset command

    def __init__(self, name: str, address: str) -> None:
        log.debug("Opening Newport_AG_UC8 at %s" % address)

        super().__init__(name,
                         address,
                         timeout=self.default_timeout,
                         terminator="\r\n")
        self.visa_handle.baud_rate = 912600

        self._current_channel: Optional[int] = None

        channels = [ Newport_AG_UC8_Channel(self, channel)
                     for channel in range(1, 4+1) ]

        channel_list = ChannelList(self,
                                   "channels",
                                   Newport_AG_UC8_Channel,
                                   channels)

        self.add_submodule("channels", channel_list)

        self.add_function("reset",
                          call_cmd=self.reset,
                          args=())

        # Set controller in remote mode (otherwise many commands don't work).
        self.write("MR")

    def get_last_error(self) -> int:
        resp = self.ask('TE')
        if resp.startswith("TE"):
            try:
                return int(resp.strip()[2:])
            except ValueError:
                pass
        log.warn("Unexpected response to TE command: %r" % resp)
        raise Newport_AG_UC8_Exception("Unexpected response to TE command: %r" % resp)

    def reset(self) -> None:
        self._current_channel = None
        super().write("RS")
        time.sleep(self.reset_delay)
        self.write("MR")

    def get_idn(self) -> Dict[str, Optional[str]]:
        resp = self.ask("VE")
        words = resp.strip().split()
        if len(words) == 2:
            model = words[0]
            version = words[1]
        else:
            log.warn("Unexpected response to VE command: %r" % resp)
            raise Newport_AG_UC8_Exception("Unexpected response to VE command: %r" % resp)
        return { "vendor": "Newport",
                 "model": model,
                 "firmware": version }

    def write(self, cmd: str) -> None:
        super().write(cmd)
        time.sleep(self.command_delay)
        err = self.get_last_error()
        if err != 0:
            log.warn("Command %s failed with error %d" % (cmd, err))
            raise Newport_AG_UC8_ErrorCode(cmd, err)

    def write_channel(self, channel: int, cmd: str) -> None:
        if channel != self._current_channel:
            super().write("CC%d" % channel)
            time.sleep(self.command_delay)
            self._current_channel = channel
        self.write(cmd)

    def ask_channel(self, channel: int, cmd: str) -> str:
        if channel != self._current_channel:
            super().write("CC%d" % channel)
            self._current_channel = channel
        return self.ask(cmd)
