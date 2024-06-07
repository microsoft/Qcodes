"""
This is a driver for the Stahl power supplies
"""

import logging
import re
from collections import OrderedDict
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from pyvisa.resources.serial import SerialInstrument
from pyvisa.resources.tcpip import TCPIPSocket

from qcodes.instrument import (
    ChannelList,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.validators import Numbers

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter

logger = logging.getLogger()


def chain(*functions: Callable[..., Any]) -> Callable[..., Any]:
    """
    The output of the first callable is piped to the input of the second, etc.

    Example:
        >>> def f():
        >>>   return "1.2"
        >>> chain(f, float)()  # return 1.2 as float
    """

    def make_iter(args: Any) -> Iterable[Any]:
        if not isinstance(args, Iterable) or isinstance(args, str):
            return (args,)
        return args

    def inner(*args: Any) -> Any:
        result = args
        for fun in functions:
            new_args = make_iter(result)
            result = fun(*new_args)

        return result

    return inner


class StahlChannel(InstrumentChannel):


    acknowledge_reply = chr(6)

    def __init__(
        self,
        parent: VisaInstrument,
        name: str,
        channel_number: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        """
        A Stahl source channel

        Args:
            parent: Parent instrument
            name: Name of the channel
            channel_number: The channel number
            **kwargs: kwargs to be passed to the base class
        """
        super().__init__(parent, name, **kwargs)

        self._channel_string = f"{channel_number:02d}"
        self._channel_number = channel_number

        _FLOATING_POINT_RE = r"[+\-]?(?:[.,]\d+|\d+(?:[.,]\d*)?)(?:[eE][-+]?\d+)?"

        self.voltage: Parameter = self.add_parameter(
            "voltage",
            get_cmd=f"{self.parent.identifier} U{self._channel_string}",
            get_parser=chain(
                re.compile(f"^({_FLOATING_POINT_RE})[ ]?V$").findall,
                partial(re.sub, ",", "."),
                float,
            ),
            set_cmd=self._set_voltage,
            unit="V",
            vals=Numbers(-self.parent.voltage_range, self.parent.voltage_range),
        )
        """Parameter voltage"""

        self.current: Parameter = self.add_parameter(
            "current",
            get_cmd=f"{self.parent.identifier} I{self._channel_string}",
            get_parser=chain(
                re.compile(f"^({_FLOATING_POINT_RE})[ ]?mA$").findall,
                partial(re.sub, ",", "."),
                lambda ma: float(ma) / 1000,  # Convert mA to A
            ),
            unit="A",
        )
        """Parameter current"""

        self.is_locked: Parameter = self.add_parameter(
            "is_locked", get_cmd=self._get_lock_status
        )
        """Parameter is_locked"""

    def _set_voltage(self, voltage: float) -> None:
        """
        Args:
            voltage: The voltage to set.
        """
        # Normalize the voltage in the range 0 to 1, where 0 is maximum negative
        # voltage and 1 is maximum positive voltage
        voltage_normalized = np.interp(
            voltage, self.parent.voltage_range * np.array([-1, 1]), [0, 1]
        )

        send_string = (
            f"{self.parent.identifier} CH{self._channel_string} "
            f"{voltage_normalized:.5f}"
        )
        response = self.ask(send_string)

        if response != self.acknowledge_reply:
            self.log.warning(
                f"Command {send_string} did not produce an acknowledge reply\n    response was: {response}"
            )

    def _get_lock_status(self) -> bool:
        """
        A lock occurs when an output is overloaded

        Return:
            lock_status: True when locked
        """
        send_string = f"{self.parent.identifier} LOCK"

        response = self.parent.visa_handle.query_binary_values(
            send_string, datatype="B", header_fmt="empty"
        )

        channel_index = self._channel_number - 1
        channel_group = channel_index // 4
        lock_code_group = response[channel_group]
        return format(lock_code_group, "b")[channel_index % 4 + 1] == "1"


class Stahl(VisaInstrument):

    default_terminator = "\r"

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ):
        """
        Stahl driver.

        Args:
            name: instrument name
            address: A serial port or TCP/IP VISA address
            **kwargs: forwarded to base class

        The TCP/IP scenario can be used when the Stahl is connected to
        a different computer, for example a Raspberry Pi running Linux,
        and exposed using something like the following script:

        ::

            #!/bin/sh
            DEVICE=/dev/ttyUSB0
            PORT=8088
            echo Listening...
            while socat $DEVICE,echo=0,b115200,raw tcp-listen:$PORT,reuseaddr,nodelay; do
                echo Restarting...
            done

        In this case the VISA address would be: ``"TCPIP0::hostname::8088::SOCKET"``
        """
        super().__init__(name, address, **kwargs)
        if isinstance(self.visa_handle, TCPIPSocket):
            pass  # allow connection to remote serial device
        elif isinstance(self.visa_handle, SerialInstrument):
            self.visa_handle.baud_rate = 115200
        else:
            raise TypeError(
                "VisaHandle must be either a 'SerialInstrument' or a 'TCPIPSocket'"
            )

        instrument_info = self.parse_idn_string(self.ask("IDN"))

        for key, value in instrument_info.items():
            setattr(self, key, value)

        channels = ChannelList(self, "channel", StahlChannel, snapshotable=False)

        for channel_number in range(1, self.n_channels + 1):
            name = f"channel{channel_number}"
            channel = StahlChannel(self, name, channel_number)
            self.add_submodule(name, channel)
            channels.append(channel)

        self.add_submodule("channel", channels)

        self.temperature: Parameter = self.add_parameter(
            "temperature",
            get_cmd=f"{self.identifier} TEMP",
            get_parser=chain(re.compile("^TEMP (.*)°C$").findall, float),
            unit="C",
        )
        """Parameter temperature"""

        self.connect_message()

    def ask_raw(self, cmd: str) -> str:
        """
        Sometimes the instrument returns non-ascii characters in response
        strings manually adjust the encoding to latin-1
        """
        self.visa_log.debug(f"Querying: {cmd}")
        self.visa_handle.write(cmd)
        response = self.visa_handle.read(encoding="latin-1")
        self.visa_log.debug(f"Response: {response}")
        return response

    @staticmethod
    def parse_idn_string(idn_string: str) -> dict[str, Any]:
        """
        Return:
             dict: The dict contains the following keys "model",
             "serial_number", "voltage_range","n_channels", "output_type"
        """
        result = re.search(r"(HV|BS)(\d{3}) (\d{3}) (\d{2}) ([buqsm])", idn_string)

        if result is None:
            raise RuntimeError(
                "Unexpected instrument response. Perhaps the model of the "
                "instrument does not match the drivers expectation or a "
                "firmware upgrade has taken place. Please get in touch "
                "with a QCoDeS core developer"
            )

        converters: dict[str, Callable[..., Any]] = OrderedDict(
            {
                "model": str,
                "serial_number": str,
                "voltage_range": float,
                "n_channels": int,
                "output_type": {
                    "b": "bipolar",
                    "u": "unipolar",
                    "q": "quadrupole",
                    "s": "steerer",
                    "m": "bipolar milivolt",
                }.get,
            }
        )

        return {
            name: converter(value)
            for (name, converter), value in zip(converters.items(), result.groups())
        }

    def get_idn(self) -> dict[str, Optional[str]]:
        """
        The Stahl sends a uncommon IDN string which does not include a
        firmware version.
        """
        return {
            "vendor": "Stahl",
            "model": self.model,
            "serial": self.serial_number,
            "firmware": None,
        }

    @property
    def identifier(self) -> str:
        return f"{self.model}{self.serial_number}"
