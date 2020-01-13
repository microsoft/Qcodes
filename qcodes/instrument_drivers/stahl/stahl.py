"""
This is a driver for the Stahl power supplies
"""

from typing import Dict, Optional, Any, Callable, Iterable
import re
import numpy as np
import logging
from collections import OrderedDict
from functools import partial

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Numbers

logger = logging.getLogger()


def chain(*functions: Callable) -> Callable:
    """
    The output of the first callable is piped to the input of the second, etc.

    Example:
        >>> def f():
        >>>   return "1.2"
        >>> chain(f, float)()  # return 1.2 as float
    """

    def make_iter(args):
        if not isinstance(args, Iterable) or isinstance(args, str):
            return args,
        return args

    def inner(*args):
        result = args
        for fun in functions:
            new_args = make_iter(result)
            result = fun(*new_args)

        return result

    return inner


class StahlChannel(InstrumentChannel):
    """
    A Stahl source channel

    Args:
        parent
        name
        channel_number
    """

    acknowledge_reply = chr(6)

    def __init__(self, parent: VisaInstrument, name: str, channel_number: int):
        super().__init__(parent, name)

        self._channel_string = f"{channel_number:02d}"
        self._channel_number = channel_number

        self.add_parameter(
            "voltage",
            get_cmd=f"{self.parent.identifier} U{self._channel_string}",
            get_parser=chain(
                re.compile(r"^([+\-]\d+,\d+) V$").findall,
                partial(re.sub, ",", "."),
                float
            ),
            set_cmd=self._set_voltage,
            unit="V",
            vals=Numbers(
                -self.parent.voltage_range,
                self.parent.voltage_range
            )
        )

        self.add_parameter(
            "current",
            get_cmd=f"{self.parent.identifier} I{self._channel_string}",
            get_parser=chain(
                re.compile(r"^([+\-]\d+,\d+) mA$").findall,
                partial(re.sub, ",", "."),
                lambda ma: float(ma) / 1000  # Convert mA to A
            ),
            unit="A",
        )

        self.add_parameter(
            "is_locked",
            get_cmd=self._get_lock_status
        )

    def _set_voltage(self, voltage: float) -> None:
        """
        Args:
            voltage
        """
        # Normalize the voltage in the range 0 to 1, where 0 is maximum negative
        # voltage and 1 is maximum positive voltage
        voltage_normalized = np.interp(
            voltage,
            self.parent.voltage_range * np.array([-1, 1]),
            [0, 1]
        )

        send_string = f"{self.parent.identifier} CH{self._channel_string} " \
            f"{voltage_normalized:.5f}"
        response = self.ask(send_string)

        if response != self.acknowledge_reply:
            self.log.warning(
                f"Command {send_string} did not produce an acknowledge reply")

    def _get_lock_status(self) -> bool:
        """
        A lock occurs when an output is overloaded

        Return:
            lock_status: True when locked
        """
        send_string = f"{self.parent.identifier} LOCK"

        response = self.parent.visa_handle.query_binary_values(
            send_string,
            datatype='B',
            header_fmt="empty"
        )

        channel_index = self._channel_number - 1
        channel_group = channel_index // 4
        lock_code_group = response[channel_group]
        return format(lock_code_group, "b")[channel_index % 4 + 1] == "1"


class Stahl(VisaInstrument):
    """
    Stahl driver.

    Args:
        name
        address: A serial port address
    """

    def __init__(self, name: str, address: str, **kwargs):
        super().__init__(name, address, terminator="\r", **kwargs)
        self.visa_handle.baud_rate = 115200

        instrument_info = self.parse_idn_string(
            self.ask("IDN")
        )

        for key, value in instrument_info.items():
            setattr(self, key, value)

        channels = ChannelList(
            self, "channel", StahlChannel, snapshotable=False
        )

        for channel_number in range(1, self.n_channels + 1):
            name = f"channel{channel_number}"
            channel = StahlChannel(
                self,
                name,
                channel_number
            )
            self.add_submodule(name, channel)
            channels.append(channel)

        self.add_submodule("channel", channels)

        self.add_parameter(
            "temperature",
            get_cmd=f"{self.identifier} TEMP",
            get_parser=chain(
                re.compile("^TEMP (.*)°C$").findall,
                float
            ),
            unit="C"
        )

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
    def parse_idn_string(idn_string) -> Dict[str, Any]:
        """
        Return:
             dict: The dict contains the following keys "model",
             "serial_number", "voltage_range","n_channels", "output_type"
        """
        result = re.search(
            r"(HV|BS)(\d{3}) (\d{3}) (\d{2}) ([buqsm])",
            idn_string
        )

        if result is None:
            raise RuntimeError(
                "Unexpected instrument response. Perhaps the model of the "
                "instrument does not match the drivers expectation or a "
                "firmware upgrade has taken place. Please get in touch "
                "with a QCoDeS core developer"
            )

        converters: Dict[str, Callable] = OrderedDict({
            "model": str,
            "serial_number": str,
            "voltage_range": float,
            "n_channels": int,
            "output_type": {
                "b": "bipolar",
                "u": "unipolar",
                "q": "quadrupole",
                "s": "steerer",
                "m": "bipolar milivolt"
            }.get
        })

        return {
            name: converter(value)
            for (name, converter), value in zip(converters.items(), result.groups())
        }

    def get_idn(self) -> Dict[str, Optional[str]]:
        """
        The Stahl sends a uncommon IDN string which does not include a
        firmware version.
        """
        return {
            "vendor": "Stahl",
            "model": self.model,
            "serial": self.serial_number,
            "firmware": None
        }

    @property
    def identifier(self):
        return f"{self.model}{self.serial_number}"
