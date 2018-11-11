"""
QCoDeS driver for Newport AGILIS AG-UC8 Piezo Stepper Controller.
"""

import logging
import time
from typing import Optional, Dict, Callable

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Ints

log = logging.getLogger(__name__)


# Meaning of error codes returned by the device.
ERROR_CODES = {
    0: "No error",
    -1: "Unknown command",
    -2: "Axis out of range",
    -3: "Wrong format for parameter",
    -4: "Parameter out of range",
    -5: "Not allowed in local mode",
    -6: "Not allowed in current state",
}


def _make_get_parser(cmd: str) -> Callable[[str], int]:
    """Return a parser function which expects a line
    starting with <cmd> followed by an integer, and returns
    only the integer.
    """

    def get_parser(resp: str) -> int:
        if resp.startswith(cmd):
            try:
                return int(resp[len(cmd):].strip())
            except ValueError:
                # Parsing return value failed.
                # Ignore the error here, we will report it below.
                pass
        log.warning("Unexpected response from %s: %r" % (cmd, resp))
        raise Newport_AG_UC8_Exception("Unexpected response from %s: %r"
                                       % (cmd, resp))

    return get_parser


class Newport_AG_UC8_Exception(Exception):
    pass


class Newport_AG_UC8_ErrorCode(Newport_AG_UC8_Exception):
    def __init__(self, cmd: str, err: int) -> None:
        self.failed_command = cmd
        self.error_code = err
        error_string = ERROR_CODES.get(err, "unknown error")
        super().__init__("Command %s failed with error code %d (%s)"
                         % (cmd, err, error_string))


class Newport_AG_UC8_Axis(InstrumentChannel):
    """Represents one of the 2 axes of one channel of an AG-UC8 controller."""

    SPEED_TABLE = {
        1: "5 steps/second at defined step amplitude",
        2: "100 steps/second at maximum step amplitude",
        3: "1700 steps/second at maximum step amplitude",
        4: "666 steps/second at defined step amplitude",
    }

    def __init__(self, parent: 'Newport_AG_UC8_Channel', axis: int) -> None:
        assert axis in (1, 2)
        super().__init__(parent, "axis_%d" % axis)

        self.axis = axis

        self.add_parameter("step_delay",
                           label="Step delay in units of 10 us",
                           get_cmd="%dDL?" % axis,
                           set_cmd="%dDL{}" % axis,
                           get_parser=_make_get_parser("%dDL" % axis),
                           vals=Ints(0, 200000))
        self.add_parameter("step_amplitude_pos",
                           label="Step amplitude in positive direction",
                           get_cmd="%dSU+?" % axis,
                           set_cmd="%dSU+{}" % axis,
                           get_parser=_make_get_parser("%dSU+" % axis),
                           vals=Ints(1, 50))
        self.add_parameter("step_amplitude_neg",
                           label="Step amplitude in negative direction",
                           get_cmd="%dSU-?" % axis,
                           set_cmd="%dSU-{}" % axis,
                           get_parser=_make_get_parser("%dSU-" % axis),
                           vals=Ints(1, 50))
        self.add_parameter("steps",
                           label="Accumulated number of steps since last "
                                 + "reset of the step counter",
                           get_cmd="%dTP" % axis,
                           get_parser=_make_get_parser("%dTP" % axis))
        self.add_parameter("status",
                           label="Status of the axis",
                           get_cmd="%dTS" % axis,
                           get_parser=_make_get_parser("%dTS" % axis),
                           val_mapping={"ready": 0,
                                        "stepping": 1,
                                        "jogging": 2,
                                        "moving_to_limit": 3})

    def jog(self, speed: int) -> None:
        """Start moving in positive (speed > 0) or negative (speed < 0)
        direction.

        Args:
            speed (int): speed and direction of movement.
                Negative values (-1 .. -4) start moving in negative direction.
                Positive values (1 .. 4) start moving in positive direction.
                Magnitude determines speed according to
                Newport_AG_UC8_Axis.SPEED_TABLE.
        """
        assert speed >= -4 and speed <= 4
        self.write("%dJA%d" % (self.axis, speed))

    def move_limit(self, speed: int) -> None:
        """Start moving to positive (speed > 0) or negative (speed < 0) limit.

        Args:
            speed (int): speed and direction of movement.
                Negative values (-1 .. -4) start moving in negative direction.
                Positive values (1 .. 4) start moving in positive direction.
                Magnitude determines speed according to
                Newport_AG_UC8_Axis.SPEED_TABLE.
        """
        assert speed >= -4 and speed <= 4
        self.write("%dMV%d" % (self.axis, speed))

    def measure_position(self) -> int:
        """Measure current position.

        This is a slow command which may take up to 2 minutes to finish.

        Returns:
            position (int): Current position in range 0 .. 1000 representing
                steps of 1/1000 of total travel.
        """
        return self._slow_command("%dMA" % self.axis, "")

    def move_abs(self, position: int) -> int:
        """Move to absolute position.

        This is a slow command which may take up to 2 minutes to finish.

        Args:
            position (int): Target position in range 0 .. 1000 representing
                steps of 1/1000 of total travel.
        """
        assert position >= 0 and position <= 1000
        return self._slow_command("%dPA" % self.axis, "%d" % position)

    def move_rel(self, steps: int) -> None:
        """Start a relative move to current position.

        Args:
            steps (int): Number of steps to move relative to current position.
        """
        assert steps >= -(2**31) and steps < 2**31
        self.write("%dPR%d" % (self.axis, steps))

    def stop(self):
        """Stop current movement."""
        self.write("%dST" % self.axis)

    def zero_position(self):
        """Reset the step counter to zero."""
        self.write("%dZP" % self.axis)

    def _slow_command(self, cmd: str, arg: str) -> int:
        """Execute a slow command with longer timeout and parse
        return value."""

        # Temporarily set long timeout to support slow command.
        tmo = self.root_instrument.slow_command_timeout
        self.root_instrument.timeout(tmo)

        try:
            # Execute command.
            resp = self.ask(cmd + arg)
        finally:
            # Restore normal timeout.
            tmo = self.root_instrument.default_timeout
            self.root_instrument.timeout(tmo)

        # Parse response.
        if resp.startswith(cmd):
            try:
                return int(resp[len(cmd):].strip())
            except ValueError:
                # Parsing return value failed.
                # Ignore the error here, we will report it below.
                pass
        log.warning("Unexpected response from %s: %r" % (cmd, resp))
        raise Newport_AG_UC8_Exception("Unexpected response from %s: %r"
                                       % (cmd, resp))


class Newport_AG_UC8_Channel(InstrumentChannel):
    """Represents one of the 4 channels of an AG-UC8 controller.

    Each channel drives 2 axes of an optical mount.
    """

    def __init__(self, parent: 'Newport_AG_UC8', channel_number: int) -> None:
        assert channel_number in (1, 2, 3, 4)
        super().__init__(parent, "channel_%d" % channel_number)

        self._channel_number = channel_number

        self.add_submodule("axis1", Newport_AG_UC8_Axis(self, 1))
        self.add_submodule("axis2", Newport_AG_UC8_Axis(self, 2))

        self.add_parameter("limit_status",
                           label="Limit switch status",
                           get_cmd="PH",
                           get_parser=_make_get_parser('PH'),
                           val_mapping={"not_active": 0,
                                        "active_on_axis_1": 1,
                                        "active_on_axis_2": 2,
                                        "active_on_both_axes": 3})

    def write(self, cmd: str) -> None:
        return self.parent.write_channel(self._channel_number, cmd)

    def ask(self, cmd: str) -> str:
        return self.parent.ask_channel(self._channel_number, cmd)


class Newport_AG_UC8(VisaInstrument):
    """
    QCoDeS driver for the Newport AGILIS AG-UC8 Piezo Stepper Controller.

    Args:
        name (str): name of the instrument.

        address (str): VISA string describing the serial port,
            for example "ASRL3" for COM3.
    """

    # By default, expect response to command within 1 second.
    default_timeout = 1.0

    # Some commands (position measurement and absolute move) can take
    # up to 2 minutes to complete.
    slow_command_timeout = 120.0

    # After a command which does not generate a response, a short
    # delay is needed before we can send the following command.
    command_delay = 0.002

    # After a reset command, a longer delay is needed before
    # we can send the following command.
    reset_delay = 0.05

    def __init__(self, name: str, address: str) -> None:
        log.debug("Opening Newport_AG_UC8 at %s" % address)

        super().__init__(name,
                         address,
                         timeout=self.default_timeout,
                         terminator="\r\n")
        self.visa_handle.baud_rate = 912600

        self._current_channel: Optional[int] = None

        channels = [Newport_AG_UC8_Channel(self, channel_number)
                    for channel_number in range(1, 4+1)]

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
        """Send a TE command (get error of previous command) and return
        a numerical error code.

        Returns:
            error (int): Error code for previous command.
                Value 0 means no error (success).
                See global table ERROR_CODES for the meaning of the codes.

        This function is called automatically after each command sent
        to the device. When a command results in error, exception
        Newport_AG_UC8_ErrorCode is raised.
        """
        resp = self.ask('TE')
        if resp.startswith("TE"):
            try:
                return int(resp.strip()[2:])
            except ValueError:
                # Parsing error code failed.
                # Ignore the error here, we will report it below.
                pass
        log.warning("Unexpected response to TE command: %r" % resp)
        raise Newport_AG_UC8_Exception("Unexpected response to TE command: %r"
                                       % resp)

    def reset(self) -> None:
        """Reset the motor controller."""
        self._current_channel = None
        # Send reset command.
        super().write("RS")
        # Sleep until reset completed.
        time.sleep(self.reset_delay)
        # Switch controller to remote mode (many commands require remote mode).
        self.write("MR")

    def get_idn(self) -> Dict[str, Optional[str]]:
        resp = self.ask("VE")
        words = resp.strip().split()
        if len(words) == 2:
            model = words[0]
            version = words[1]
        else:
            log.warning("Unexpected response to VE command: %r" % resp)
            raise Newport_AG_UC8_Exception(
                "Unexpected response to VE command: %r" % resp)
        return {"vendor": "Newport",
                "model": model,
                "firmware": version}

    def write(self, cmd: str) -> None:
        # Send command.
        super().write(cmd)
        # Sleep until command completed.
        time.sleep(self.command_delay)
        # Check if command produced an error.
        err = self.get_last_error()
        if err != 0:
            log.warning("Command %s failed with error %d" % (cmd, err))
            raise Newport_AG_UC8_ErrorCode(cmd, err)

    def _select_channel(self, channel_number: int) -> None:
        """Make sure the specified channel is selected."""
        if self._current_channel != channel_number:
            # Switch to channel.
            super().write("CC%d" % channel_number)
            # Sleep until change channel command completed.
            time.sleep(self.command_delay)
            self._current_channel = channel_number

    def write_channel(self, channel_number: int, cmd: str) -> None:
        """Select specified channel, then apply specified command."""
        self._select_channel(channel_number)
        self.write(cmd)

    def ask_channel(self, channel_number: int, cmd: str) -> str:
        """Select specified channel, then apply specified query
        and return response."""
        self._select_channel(channel_number)
        return self.ask(cmd)
