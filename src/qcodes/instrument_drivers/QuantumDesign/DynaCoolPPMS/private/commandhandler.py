import logging
from collections import namedtuple
from itertools import chain
from typing import Any, ClassVar

log = logging.getLogger(__name__)

try:
    import pythoncom  # pyright: ignore[reportMissingModuleSource]
    import win32com.client  # pyright: ignore[reportMissingModuleSource]
    from pythoncom import (  # pyright: ignore[reportMissingModuleSource]
        VT_BYREF,
        VT_I4,
        VT_R8,
    )
except ImportError as e:
    message = "To use the DynaCool Driver, please install pywin32."
    log.exception(message)
    raise ImportError(message) from e


CmdArgs = namedtuple("CmdArgs", "cmd args")

# The length of a command header, aka a command keyword
# Every command sent from the driver via the server must have a
# keyword of exactly this length ('?' NOT included)
CMD_HEADER_LENGTH = 4


class CommandHandler:
    """
    This is the class that gets called by the server.py

    This class is responsible for making the actual calls into the instrument
    firmware. The idea is that this class get a SCPI-like string from the
    server, e.g. 'TEMP?' or 'TEMP 300, 10, 1' and then makes the corresponding
    MultiVu API call (or returns an error message to the server).
    """

    # Variable types
    _variants: ClassVar[dict[str, win32com.client.VARIANT]] = {
        "double": win32com.client.VARIANT(VT_BYREF | VT_R8, 0.0),
        "long": win32com.client.VARIANT(VT_BYREF | VT_I4, 0),
    }

    def __init__(self, inst_type: str = "dynacool") -> None:
        self.inst_type = inst_type
        pythoncom.CoInitialize()
        client_id = f"QD.MULTIVU.{inst_type.upper()}.1"
        try:
            self._mvu = win32com.client.Dispatch(client_id)
        except pythoncom.com_error:
            error_mssg = (
                "Could not connect to Multivu Application. Please "
                "make sure that the MultiVu Application is running."
            )
            log.exception(error_mssg)
            raise ValueError(error_mssg)

        _variants = CommandHandler._variants

        # Hard-code what we know about the MultiVu API
        self._gets = {
            "TEMP": CmdArgs(
                cmd=self._mvu.GetTemperature,
                args=[_variants["double"], _variants["long"]],
            ),
            "CHAT": CmdArgs(
                cmd=self._mvu.GetChamberTemp,
                args=[_variants["double"], _variants["long"]],
            ),
            "GLTS": CmdArgs(
                cmd=self._mvu.GetLastTempSetpoint,
                args=[_variants["double"], _variants["double"], _variants["long"]],
            ),
            "GLFS": CmdArgs(
                cmd=self._mvu.GetFieldSetpoints,
                args=[
                    _variants["double"],
                    _variants["double"],
                    _variants["long"],
                    _variants["long"],
                ],
            ),
            "CHAM": CmdArgs(cmd=self._mvu.GetChamber, args=[_variants["long"]]),
            "FELD": CmdArgs(
                cmd=self._mvu.GetField, args=[_variants["double"], _variants["long"]]
            ),
            "*IDN": CmdArgs(cmd=self.make_idn_string, args=[]),
        }

        self._sets = {"TEMP": self._mvu.SetTemperature, "FELD": self._mvu.SetField}

        # validate the commands
        for cmd in chain(self._gets, self._sets):
            if len(cmd) != CMD_HEADER_LENGTH:
                raise ValueError(
                    f"Invalid command length: {cmd}."
                    f" Must have length {CMD_HEADER_LENGTH}"
                )

    def make_idn_string(self) -> str:
        return f"0, QuantumDesign, {self.inst_type}, N/A, N/A"

    def preparser(self, cmd_str: str) -> tuple[CmdArgs, bool]:
        """
        Parse the raw SCPI-like input string into a CmdArgs tuple containing
        the corresponding MultiVu API function and a boolean indicating whether
        we expect the MultiVu function to modify its input (i.e. be a query)

        Args:
            cmd_str: A SCPI-like string, e.g. 'TEMP?' or 'TEMP 300, 0.1, 1'

        Returns:
            A tuple of a CmdArgs tuple and a bool indicating whether this was
            a query
        """

        def err_func() -> int:
            return -2

        cmd_head = cmd_str[:CMD_HEADER_LENGTH]

        if cmd_head not in set(self._gets.keys()).union(set(self._sets.keys())):
            cmd = err_func
            args: list[Any] = []
            is_query = False

        elif cmd_str.endswith("?"):
            cmd = self._gets[cmd_head].cmd
            args = self._gets[cmd_head].args
            is_query = True
        else:
            cmd = self._sets[cmd_head]
            args = list(float(arg) for arg in cmd_str[5:].split(", "))
            is_query = False

        return CmdArgs(cmd=cmd, args=args), is_query

    @staticmethod
    def postparser(error_code: int, vals: list[Any]) -> str:
        """
        Parse the output of the MultiVu API call into a string that the server
        can send back to the client

        Args:
            error_code: the error code returned from the MultiVu call
            vals: A list of the returned values (empty in case of a set cmd)
        """
        response = f"{error_code}"

        for val in vals:
            response += f", {val}"

        return response

    def __call__(self, cmd: str) -> str:
        cmd_and_args, is_query = self.preparser(cmd)
        log.debug(f"Parsed {cmd} into {cmd_and_args}")

        # Actually perform the call into the MultiVu API
        error_code = cmd_and_args.cmd(*cmd_and_args.args)

        # return values in case we did a query
        if is_query:
            # read out the mutated values
            # (win32 reverses the order)
            vals = list(arg.value for arg in cmd_and_args.args)
            vals.reverse()
            # reset the value variables for good measures
            for arg in cmd_and_args.args:
                arg.value = 0
        else:
            vals = []

        response_message = self.postparser(error_code, vals)

        return response_message
